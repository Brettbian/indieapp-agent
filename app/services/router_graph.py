"""LangGraph-based router for handling chat and generation requests.

This module implements a unified RouterGraph that replaces procedural routing
with LangGraph's conditional nodes and built-in checkpointer for memory persistence.
"""

import json
import re
import uuid
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig

from app.core.logging import get_logger
from app.services.llm_service import LLMService
from app.services.supervisor_agent import SupervisorAgent, RequestType
from app.services.chat_agent import ChatAgent
from app.services.s3_service import get_s3_service
from app.services.tasks import generate_content
from app.services.compaction_node import compaction_node, COMPACTION_SUMMARY_PREFIX
from app.core.config import settings

logger = get_logger(__name__)


class RouterState(MessagesState):
    """State for RouterGraph.

    MessagesState provides: messages: Annotated[list[AnyMessage], add_messages]
    - messages: PERSISTS across invocations (conversation history via add_messages reducer)
    - All other fields: RESET each invocation (inter-node communication)

    When the graph is invoked, pass input_state with:
    - messages: [HumanMessage(content=user_prompt)]  # Auto-merged with history
    - All other fields reset to defaults
    """

    # Inter-node communication fields (reset each invocation via input_state)
    request_type: Optional[str] = None  # "chat" or "generation"
    classification_reasoning: str = ""
    is_context_sufficient: bool = True
    clarifying_questions: str = ""
    filter_intent_unclear: bool = False  # Set when filter LLM returns empty (e.g., safety blocked)
    planned_tasks: List[Dict[str, str]] = []
    celery_task_ids: List[str] = []  # List of task IDs (serializable)
    task_map: Dict[str, str] = {}  # task_id -> title mapping
    task_type_map: Dict[str, str] = {}  # task_id -> type mapping
    task_return_type_map: Dict[str, str] = {}  # task_id -> return file_type (inherits from attachment)
    error: Optional[str] = None


def _format_messages_for_routing(messages: List) -> str:
    """Format messages for routing context.

    Args:
        messages: List of LangChain message objects

    Returns:
        Formatted string for routing LLM
    """
    if not messages:
        return ""

    conversation_context = "Recent conversation:\n"
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        conversation_context += f"{role}: {content}\n"

    return conversation_context


def _format_conversation_context(messages: List) -> str:
    """Format messages as conversation context for task planning.

    Args:
        messages: List of LangChain message objects

    Returns:
        Formatted conversation string
    """
    if not messages:
        return ""

    context_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith(COMPACTION_SUMMARY_PREFIX) or content.startswith("[Task Completed]"):
                role = "system"
            else:
                continue  # Skip other system messages
        else:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        context_parts.append(f"{role}: {content}")

    return "\n".join(context_parts)


async def classify_request(state: RouterState, config: RunnableConfig) -> dict:
    """Classify the request as CHAT or GENERATION.

    Uses the last 8 messages for routing context.

    Args:
        state: Current router state with messages
        config: Runtime config with thread_id, request_id, attachments

    Returns:
        Dict with request_type and classification_reasoning
    """
    request_id = config["configurable"].get("request_id", "unknown")

    # Get last 8 messages for routing context
    recent_msgs = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=8,
        token_counter=len,  # Count messages, not tokens
    )

    # Format conversation history for routing
    conversation_context = _format_messages_for_routing(recent_msgs[:-1])  # Exclude current

    # Get the current user prompt (last message)
    current_message = state["messages"][-1] if state["messages"] else None
    user_prompt = current_message.content if current_message else ""

    # Combine context with current prompt
    if conversation_context:
        prompt_with_context = f"{conversation_context}\nCurrent request: {user_prompt}"
    else:
        prompt_with_context = user_prompt

    # Use supervisor to classify
    supervisor = SupervisorAgent()
    request_type, reasoning = await supervisor.analyze_request(prompt_with_context)

    logger.info(
        "Request classified",
        request_id=request_id,
        request_type=request_type.value,
        reasoning=reasoning,
    )

    return {
        "request_type": request_type.value,
        "classification_reasoning": reasoning,
    }


async def validate_generation_context(state: RouterState, config: RunnableConfig) -> dict:
    """Validate if we have sufficient context for generation.

    Uses the last 50 messages for validation context.

    Args:
        state: Current router state
        config: Runtime config with attachments

    Returns:
        Dict with is_context_sufficient and clarifying_questions
    """
    request_id = config["configurable"].get("request_id", "unknown")
    attachments = config["configurable"].get("attachments", [])

    # Get last 50 messages for validation
    recent_msgs = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=50,
        token_counter=len,
    )

    # Get current user prompt
    current_message = state["messages"][-1] if state["messages"] else None
    user_prompt = current_message.content if current_message else ""

    # Build context summary
    context_summary = []

    if len(recent_msgs) > 1:
        context_summary.append(f"Conversation history: {len(recent_msgs)} messages")

    if attachments:
        attachment_types = [att.get("type", "document") for att in attachments]
        context_summary.append(
            f"Attachments: {len(attachments)} files ({', '.join(attachment_types)})"
        )

    # Check if we've recently asked clarifying questions
    recent_assistant_asked_questions = False
    for msg in recent_msgs[-6:]:
        if isinstance(msg, AIMessage):
            content = msg.content.lower() if isinstance(msg.content, str) else ""
            if any(
                phrase in content
                for phrase in [
                    "need a bit more information",
                    "please provide",
                    "could you clarify",
                    "what type",
                    "which",
                ]
            ):
                recent_assistant_asked_questions = True
                break

    # Quick check for common "just proceed" phrases
    user_prompt_lower = user_prompt.lower().strip()
    proceed_phrases = [
        "go ahead", "generate it", "create it", "build it", "make it",
        "yes", "ok", "okay", "sure", "sounds good", "that's fine",
        "just do it", "proceed", "continue", "let's go",
    ]

    # If user said a proceed phrase after we asked questions, skip validation
    if recent_assistant_asked_questions and any(
        phrase in user_prompt_lower for phrase in proceed_phrases
    ):
        logger.info(
            "User used proceed phrase after clarifying questions, skipping validation",
            request_id=request_id,
        )
        return {"is_context_sufficient": True, "clarifying_questions": ""}

    # Use LLM to validate context sufficiency
    validation_prompt = (
        """You are a helpful co-pilot that enables users to create quickly.

User Request: """
        + user_prompt
        + """

Available Context:
"""
        + "\n".join(context_summary)
        + f"""

{"The assistant has ALREADY asked clarifying questions. The user's response indicates they want to proceed. Default to SUFFICIENT." if recent_assistant_asked_questions else ""}

Mark as SUFFICIENT (default) if:
- User has provided ANY reasonable request that can be acted upon
- User has answered previous questions (even briefly)
- User uses phrases like "go ahead", "yes", "create it", "build it"
- The request has at least a basic idea of what they want

Only mark as INSUFFICIENT if the request is COMPLETELY IMPOSSIBLE to act on:
- The request is literally empty or just "hello"
- The request is completely nonsensical
- CRITICAL information is missing (e.g., "edit it" with no context about what to edit)

When insufficient, ask ONLY 1 clarifying question about what they want to create.

Respond with ONLY a JSON object:
{{
  "is_sufficient": true or false,
  "clarifying_questions": "ONE brief question if truly impossible to proceed"
}}"""
    )

    try:
        validator_llm = LLMService.get_instance(component="validator")
        messages = LLMService.create_messages(
            system_prompt="You are a helpful co-pilot that empowers users to create. Default to 'sufficient' unless truly impossible. Be an enabler, not a blocker.",
            user_prompt=validation_prompt,
        )

        response = await validator_llm.generate(messages=messages)

        # Parse JSON response
        response_text = LLMService.extract_json_from_response(response)
        result = json.loads(response_text)

        is_sufficient = result.get("is_sufficient", True)
        clarifying_questions = result.get("clarifying_questions", "")

        logger.info(
            "Context validation completed",
            request_id=request_id,
            is_sufficient=is_sufficient,
        )

        return {
            "is_context_sufficient": is_sufficient,
            "clarifying_questions": clarifying_questions,
        }

    except Exception as e:
        logger.warning(
            "Context validation failed, proceeding with generation",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        # Default to sufficient if validation fails
        return {"is_context_sufficient": True, "clarifying_questions": ""}


async def filter_attachments_by_intent(state: RouterState, config: RunnableConfig) -> dict:
    """Mark attachments with should_edit flag based on user intent.

    For ppt_png attachments, parses user prompt to determine which pages
    the user wants to edit. All pages are kept, but only target pages
    are marked with should_edit=True.

    Args:
        state: Current router state
        config: Runtime config with attachments

    Returns:
        Empty dict (modifies attachments in place)
    """
    request_id = config["configurable"].get("request_id", "unknown")
    attachments = config["configurable"].get("attachments", [])

    # Only process if we have multiple ppt_png attachments
    ppt_png_attachments = [att for att in attachments if att.get("type") == "ppt_png"]

    if len(ppt_png_attachments) <= 1:
        # Single or no ppt_png: mark all as should_edit=True (default behavior)
        for att in attachments:
            att["should_edit"] = True
        logger.info(
            "Single or no ppt_png attachments - marking all as should_edit=True",
            request_id=request_id,
            total_attachments=len(attachments),
            ppt_png_count=len(ppt_png_attachments),
        )
        return {}

    # Get current user prompt
    current_message = state["messages"][-1] if state["messages"] else None
    user_prompt = current_message.content if current_message else ""

    if not user_prompt:
        # No prompt: mark all as should_edit=True
        for att in attachments:
            att["should_edit"] = True
        logger.info(
            "No user prompt - marking all as should_edit=True",
            request_id=request_id,
        )
        return {}

    try:
        # Extract page numbers/indices from attachment titles
        # Title format: "PPT Name (1/5)", "PPT Name (2/5)", etc.
        attachment_info = []
        for i, att in enumerate(ppt_png_attachments):
            title = att.get("title", "")
            # Extract page number from title like "History of China (1/5)"
            match = re.search(r'\((\d+)/(\d+)\)', title)
            if match:
                page_num = int(match.group(1))
                total_pages = int(match.group(2))
                attachment_info.append({
                    "index": i,
                    "page_num": page_num,
                    "total_pages": total_pages,
                    "title": title,
                    "attachment": att,
                })

        if not attachment_info:
            # Cannot parse page numbers: mark all as should_edit=True
            for att in attachments:
                att["should_edit"] = True
            logger.info(
                "Cannot parse page numbers from titles - marking all as should_edit=True",
                request_id=request_id,
            )
            return {}

        # Use LLM to parse user intent with role categorization
        llm_service = LLMService.get_instance(component="router")

        # Build dynamic page list for the prompt
        total_pages = attachment_info[0]["total_pages"] if attachment_info else 5
        all_pages_list = list(range(1, total_pages + 1))

        filter_prompt = f"""Analyze the user's request and determine which PPT pages should be edited.

User Request: {user_prompt}

Available PPT Pages:
{chr(10).join([f"  Page {info['page_num']}: {info['title']}" for info in attachment_info])}

Determine which pages the user wants to EDIT (modify/change).
- EDIT: Pages that should be edited/modified based on the user's request
- REFERENCE: Pages mentioned as style/format references (NOT edited, just used as examples)

Examples (English and Chinese):
1. "edit page 2" OR "修改第二页" → {{"edit": [2], "reference": []}}
2. "modify page 2 using page 3 as reference" OR "按照第三页的格式修改第二页" → {{"edit": [2], "reference": [3]}}
3. "edit pages 2 and 3 in the style of page 1" OR "用第一页的风格编辑第二和第三页" → {{"edit": [2, 3], "reference": [1]}}
4. "edit all pages" OR "编辑全部" → {{"edit": "all", "reference": []}}
5. "only edit first page" OR "只编辑第一页" → {{"edit": [1], "reference": []}}

CRITICAL: Return ONLY a valid JSON object:
{{"edit": [page_numbers] or "all", "reference": [page_numbers]}}

If the user doesn't specify which pages, or wants to edit all, return:
{{"edit": "all", "reference": []}}

Your response (JSON only):"""

        # Use generate() with proper messages format
        messages = LLMService.create_messages(
            system_prompt="You are a JSON parser that extracts page numbers from user requests. Return only valid JSON.",
            user_prompt=filter_prompt,
        )
        response_text = await llm_service.generate(
            messages=messages,
            temperature=0.0,
        )
        response_text = response_text.strip()

        logger.info(
            "LLM filter response",
            request_id=request_id,
            user_prompt=user_prompt,
            llm_response=response_text,
        )

        # Check for empty response (likely safety filter blocked)
        if not response_text:
            logger.warning(
                "LLM filter returned empty response (likely safety blocked), interrupting generation",
                request_id=request_id,
                user_prompt=user_prompt[:100],
            )
            # Set flag to interrupt generation and ask for clarification
            return {
                "filter_intent_unclear": True,
                "clarifying_questions": "I cannot determine which page you want to edit. Your input may contain some special phrasing, please try describing it in a simpler and more direct way.",
            }

        # Parse JSON response
        try:
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")

            result = json.loads(json_match.group(0))
            edit_pages = result.get("edit", [])
            reference_pages = result.get("reference", [])

            # Determine which pages to edit
            if edit_pages == "all":
                edit_page_set = set(all_pages_list)
            elif isinstance(edit_pages, list):
                edit_page_set = set(edit_pages)
            else:
                edit_page_set = set(all_pages_list)  # Default to all

            reference_page_set = set(reference_pages) if isinstance(reference_pages, list) else set()

            # If no edit pages identified, default to all
            if not edit_page_set:
                edit_page_set = set(all_pages_list)
                logger.warning(
                    "No edit pages identified, defaulting to all pages",
                    request_id=request_id,
                )

            # Mark all attachments with should_edit flag
            # Keep ALL pages, just mark which ones should be edited
            for info in attachment_info:
                page_num = info["page_num"]
                att = info["attachment"]
                att["should_edit"] = (page_num in edit_page_set)

            # Mark non-ppt_png attachments as should_edit=True (default behavior)
            for att in attachments:
                if att.get("type") != "ppt_png":
                    att["should_edit"] = True

            # Count for logging
            edit_count = sum(1 for info in attachment_info if info["attachment"].get("should_edit"))
            pass_through_count = len(attachment_info) - edit_count

            logger.info(
                "Attachment marking completed",
                request_id=request_id,
                total_ppt_png_count=len(ppt_png_attachments),
                edit_count=edit_count,
                pass_through_count=pass_through_count,
                edit_pages=sorted(edit_page_set),
                reference_pages=sorted(reference_page_set),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # If JSON parsing fails, mark all as should_edit=True
            logger.warning(
                "Failed to parse JSON response, marking all as should_edit=True",
                request_id=request_id,
                llm_response=response_text,
                error=str(e),
            )
            for att in attachments:
                att["should_edit"] = True

    except Exception as e:
        # On any error, mark all as should_edit=True (safe default)
        logger.warning(
            "Attachment filtering failed, marking all as should_edit=True",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        for att in attachments:
            att["should_edit"] = True

    return {}


async def plan_generation_tasks(state: RouterState, config: RunnableConfig) -> dict:
    """Plan what artifacts to generate.

    Uses the last 8 messages for context.

    Args:
        state: Current router state
        config: Runtime config with attachments

    Returns:
        Dict with planned_tasks or error
    """
    request_id = config["configurable"].get("request_id", "unknown")
    attachments = config["configurable"].get("attachments", [])

    # Get last 8 messages for context
    recent_msgs = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=8,
        token_counter=len,
    )

    # Get current user prompt
    current_message = state["messages"][-1] if state["messages"] else None
    user_prompt = current_message.content if current_message else ""

    # Format conversation context (exclude current message)
    conversation_context = _format_conversation_context(recent_msgs[:-1])

    try:
        supervisor = SupervisorAgent()
        planned_tasks = await supervisor.plan_generation_tasks(
            user_prompt, conversation_context, attachments
        )

        logger.info(
            "Tasks planned",
            request_id=request_id,
            task_count=len(planned_tasks),
        )

        return {"planned_tasks": planned_tasks, "error": None}

    except Exception as e:
        logger.error(
            "Task planning failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )

        # Build error message
        error_msg = f"I encountered an error while planning your request: {str(e)}"
        if (
            "content_filter" in str(e).lower()
            or "content management policy" in str(e).lower()
        ):
            error_msg = "I'm unable to plan this generation request due to content policy restrictions. Please rephrase your request or remove any potentially sensitive content from the conversation."
        elif "400" in str(e) and "bad request" in str(e).lower():
            error_msg = "I encountered an issue while analyzing your generation request. This might be due to content restrictions. Please try rephrasing your request."

        return {"planned_tasks": [], "error": error_msg}


async def dispatch_chat(state: RouterState, config: RunnableConfig) -> dict:
    """Dispatch to ChatAgent for Q&A responses.

    Issue 8 fix: uses 20 messages and aligns with ChatAgent's internal limit.
    Issue 9 fix: includes compaction summary SystemMessages in conversation history.

    Args:
        state: Current router state
        config: Runtime config with attachments, request_id

    Returns:
        Dict with AI message to append
    """
    request_id = config["configurable"].get("request_id", "unknown")
    attachments = config["configurable"].get("attachments", [])

    # Issue 8 fix: 10 -> 20 (aligned with ChatAgent.create_agent_graph[-20:])
    recent_msgs = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=20,
        token_counter=len,
    )

    # Get current user prompt
    current_message = state["messages"][-1] if state["messages"] else None
    user_prompt = current_message.content if current_message else ""

    # Format conversation history for ChatAgent (exclude current message)
    # Issue 9 fix: include compaction summary and task-result SystemMessages
    conversation_history = []
    for msg in recent_msgs[:-1]:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.startswith(COMPACTION_SUMMARY_PREFIX) or content.startswith("[Task Completed]"):
                conversation_history.append({"role": "system", "content": content})

    # Create chat agent and get response
    chat_agent = ChatAgent()
    response = await chat_agent.chat(
        user_prompt=user_prompt,
        attachments=attachments if attachments else None,
        conversation_history=conversation_history if conversation_history else None,
        request_id=request_id,
    )

    logger.info(
        "Chat response generated",
        request_id=request_id,
        response_length=len(response),
    )

    # Return AI message - will be auto-appended to messages via add_messages reducer
    return {"messages": [AIMessage(content=response)]}


async def dispatch_generation(state: RouterState, config: RunnableConfig) -> dict:
    """Dispatch generation tasks to Celery workers.

    Args:
        state: Current router state with planned_tasks
        config: Runtime config with thread_id, request_id, attachments

    Returns:
        Dict with celery_tasks, task_map, task_type_map, and intro message
    """
    request_id = config["configurable"].get("request_id", "unknown")
    thread_id = config["configurable"].get("thread_id", "")
    attachments = config["configurable"].get("attachments", [])

    # Get last 8 messages for conversation context
    recent_msgs = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=8,
        token_counter=len,
    )
    conversation_context = _format_conversation_context(recent_msgs[:-1])

    # Create tasks and dispatch to Celery
    task_map = {}
    task_type_map = {}
    task_return_type_map = {}  # For preserving original attachment type in return
    celery_task_ids = []

    for task_plan in state["planned_tasks"]:
        task_id = str(uuid.uuid4())
        task_map[task_id] = task_plan["title"]
        task_type = task_plan.get("type", "webapp")
        task_type_map[task_id] = task_type

        # Determine return file_type - inherit from attachment for image_edit
        return_type = task_type
        if task_type == "image_edit" and attachments:
            # Check if the attachment is ppt_png - if so, preserve that type
            for att in attachments:
                att_type = att.get("type", "") if isinstance(att, dict) else getattr(att, "type", "")
                if att_type == "ppt_png":
                    return_type = "ppt_png"
                    break
                elif att_type.startswith("image"):
                    return_type = "image"
                    # Don't break - keep checking for ppt_png which takes priority
        task_return_type_map[task_id] = return_type

        # Submit task to Celery
        generate_content.apply_async(
            args=[
                task_id,
                thread_id,  # project_id
                request_id,
                attachments,  # Already in dict format
                task_plan.get("description", ""),
                task_plan.get("type", "webapp"),
                conversation_context,
            ],
            task_id=task_id,
        )

        logger.info(
            "Celery task created",
            request_id=request_id,
            celery_task_id=task_id,
            task_type=task_plan.get("type", "webapp"),
        )

        celery_task_ids.append(task_id)

    # Generate intro message
    intro_message = await _generate_intro_message(state["planned_tasks"], request_id)

    return {
        "celery_task_ids": celery_task_ids,
        "task_map": task_map,
        "task_type_map": task_type_map,
        "task_return_type_map": task_return_type_map,
        "messages": [AIMessage(content=intro_message)],
    }


async def _generate_intro_message(
    planned_tasks: List[Dict[str, str]], request_id: str
) -> str:
    """Generate an intelligent intro message for generation tasks.

    Args:
        planned_tasks: List of planned task dictionaries
        request_id: Request ID for logging

    Returns:
        Natural intro message
    """
    try:
        # Prepare task summary
        task_summary = []
        for task in planned_tasks:
            task_type = task.get("type", "content")
            title = task.get("title", "")
            task_summary.append(f"- {title} ({task_type})")

        task_list_str = "\n".join(task_summary)

        # Detect if any task is an edit request
        is_edit_request = any(task.get("type") == "image_edit" for task in planned_tasks)

        if is_edit_request:
            system_prompt = """You are a helpful AI assistant about to modify/edit content for the user.
Create a brief, natural, and informative response that acknowledges what you're about to modify.
Be specific about what you're editing but keep it concise (1-2 sentences).
Use a friendly, professional tone. Don't use bullet points or lists.
IMPORTANT: Use words like "modify", "edit", "update", or "change" - NOT "generate" or "create new"."""
        else:
            system_prompt = """You are a helpful AI assistant about to generate content for the user.
Create a brief, natural, and informative response that acknowledges what you're about to create.
Be specific about what you're generating but keep it concise (1-2 sentences).
Use a friendly, professional tone. Don't use bullet points or lists.
Focus on the value and purpose of what you're creating."""

        user_prompt = f"""Generate a brief intro message for the following planned tasks:
{task_list_str}

Respond with ONLY the intro message, no JSON or extra formatting."""

        llm_service = LLMService.get_instance(component="router")
        messages = LLMService.create_messages(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        response = await llm_service.generate(messages=messages)
        return response.strip()

    except Exception as e:
        logger.warning(
            "Failed to generate intro message, using default",
            request_id=request_id,
            error=str(e),
        )
        # Default fallback
        if len(planned_tasks) == 1:
            return f"I'm working on creating {planned_tasks[0].get('title', 'your content')} for you."
        else:
            return f"I'm working on creating {len(planned_tasks)} items for you."


async def return_clarifying_questions(state: RouterState, config: RunnableConfig) -> dict:
    """Return clarifying questions when context is insufficient.

    Args:
        state: Current router state with clarifying_questions
        config: Runtime config

    Returns:
        Dict with AI message containing clarifying questions
    """
    request_id = config["configurable"].get("request_id", "unknown")

    clarifying_message = (
        f"I'd love to help you generate this! To create exactly what you're looking for, "
        f"I need a bit more information:\n\n{state['clarifying_questions']}\n\n"
        f"Please provide these details so I can generate the perfect solution for you."
    )

    logger.info(
        "Returning clarifying questions",
        request_id=request_id,
    )

    return {"messages": [AIMessage(content=clarifying_message)]}


async def return_planning_error(state: RouterState, config: RunnableConfig) -> dict:
    """Return error message when planning fails.

    Args:
        state: Current router state with error
        config: Runtime config

    Returns:
        Dict with AI message containing error
    """
    request_id = config["configurable"].get("request_id", "unknown")

    logger.info(
        "Returning planning error",
        request_id=request_id,
        error=state["error"],
    )

    return {"messages": [AIMessage(content=state["error"])]}


# Routing functions


def route_after_classification(state: RouterState) -> str:
    """Route based on classification result.

    Args:
        state: Current router state

    Returns:
        Next node name: "dispatch_chat" or "validate_context"
    """
    if state["request_type"] == RequestType.CHAT.value:
        return "dispatch_chat"
    else:
        return "validate_context"


def route_after_validation(state: RouterState) -> str:
    """Route based on validation result.

    Args:
        state: Current router state

    Returns:
        Next node name: "filter_attachments" or "return_clarifying_questions"
    """
    if state["is_context_sufficient"]:
        return "filter_attachments"
    else:
        return "return_clarifying_questions"


def route_after_filter(state: RouterState) -> str:
    """Route based on filter result.

    If the filter LLM returned empty (e.g., safety blocked), we interrupt
    and ask for clarification instead of proceeding with incorrect edits.

    Args:
        state: Current router state

    Returns:
        Next node name: "plan_tasks" or "return_clarifying_questions"
    """
    if state.get("filter_intent_unclear", False):
        return "return_clarifying_questions"
    else:
        return "plan_tasks"


def route_after_planning(state: RouterState) -> str:
    """Route based on planning result.

    Args:
        state: Current router state

    Returns:
        Next node name: "dispatch_generation" or "return_planning_error"
    """
    if state["error"]:
        return "return_planning_error"
    else:
        return "dispatch_generation"


def create_router_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Create and compile the router graph.

    Args:
        checkpointer: Optional checkpointer for message persistence.
                     If provided, messages will automatically persist across invocations.

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(RouterState)

    # Add nodes
    graph.add_node("compaction", compaction_node)
    graph.add_node("classify_request", classify_request)
    graph.add_node("validate_context", validate_generation_context)
    graph.add_node("filter_attachments", filter_attachments_by_intent)
    graph.add_node("plan_tasks", plan_generation_tasks)
    graph.add_node("dispatch_chat", dispatch_chat)
    graph.add_node("dispatch_generation", dispatch_generation)
    graph.add_node("return_clarifying_questions", return_clarifying_questions)
    graph.add_node("return_planning_error", return_planning_error)

    # Add edges — compaction runs first, then classification
    graph.add_edge(START, "compaction")
    graph.add_edge("compaction", "classify_request")

    graph.add_conditional_edges(
        "classify_request",
        route_after_classification,
        {
            "dispatch_chat": "dispatch_chat",
            "validate_context": "validate_context",
        },
    )

    graph.add_conditional_edges(
        "validate_context",
        route_after_validation,
        {
            "filter_attachments": "filter_attachments",
            "return_clarifying_questions": "return_clarifying_questions",
        },
    )

    # Filter may interrupt if LLM returned empty (safety blocked)
    graph.add_conditional_edges(
        "filter_attachments",
        route_after_filter,
        {
            "plan_tasks": "plan_tasks",
            "return_clarifying_questions": "return_clarifying_questions",
        },
    )

    graph.add_conditional_edges(
        "plan_tasks",
        route_after_planning,
        {
            "dispatch_generation": "dispatch_generation",
            "return_planning_error": "return_planning_error",
        },
    )

    # Terminal edges
    graph.add_edge("dispatch_chat", END)
    graph.add_edge("dispatch_generation", END)
    graph.add_edge("return_clarifying_questions", END)
    graph.add_edge("return_planning_error", END)

    # Compile with optional checkpointer
    return graph.compile(checkpointer=checkpointer)
