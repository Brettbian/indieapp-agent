"""LangChain agent for intelligent chat responses with document access."""

from datetime import datetime
from typing import List, Dict, Any, AsyncIterator, Optional
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.services.agent_utils import invoke_agent_with_streaming
from app.services.llm_service import LLMService
from app.services.langchain_s3_tools import S3ContextReadTool, S3ListTool
from app.services.langchain_image_tools import ImageAnalysisTool
from app.services.langchain_search_tool import GoogleSearchTool
from app.services.s3_service import get_s3_service

from app.core.config import settings, LLMProvider
from app.core.logging import get_logger
from app.core.logging_utils import log_timing

logger = get_logger(__name__)


CHAT_SYSTEM_PROMPT = """You are an intelligent thinking copilot designed to be a proactive partner in problem-solving and analysis.

You have access to:
1. **Google Search** - search the web for current information, news, real-time data
2. S3 tools to read documents when needed (only if document paths are provided)
3. Image analysis tool to analyze images from URLs
4. Previous conversation history for context

RESPONSE STYLE - CRITICAL:
- Be CONCISE and DIRECT - aim for brevity without sacrificing clarity
- Get to the point quickly in the first 1-2 sentences
- Use bullet points or numbered lists when presenting multiple items
- Avoid lengthy preambles, excessive explanations, or redundant phrases
- Keep responses under 300 words unless the user explicitly asks for detailed explanations
- Skip pleasantries like "I'd be happy to help" - jump straight to the answer

TECHNICAL WRITING STYLE:
- Use clear, precise technical language
- Include specific examples when explaining concepts
- Reference documentation or best practices where relevant
- Provide actionable insights and recommendations

GOOGLE SEARCH - WHEN TO USE (CRITICAL):
You have access to Google Search and SHOULD use it for:
- **Current date/time questions**: "今天几号", "what date is it", "今天是星期几"
- **Real-time data**: stock prices, exchange rates, weather, sports scores
- **Recent news/events**: "latest news about X", "今天的新闻", "2026年发生了什么"
- **Current facts**: pricing, availability, latest version of software
- **When user explicitly asks to search**: "搜索一下", "帮我查查", "search for", "look up"

DO NOT refuse to search by saying "I cannot access the internet" - you CAN and SHOULD use Google Search for these queries.

IMPORTANT - When you USE Google Search:
- **ALWAYS start your response with**: "根据 Google 搜索结果（[current date]）：" or "According to Google Search results ([current date]):"
- This lets users know you actually searched the internet
- Example: "根据 Google 搜索结果（[current date]）：今天是星期二..."

Example responses:
- User: "今天几号？" → Use Google Search → Start with "根据 Google 搜索结果（[current date]）：..."
- User: "日元汇率" → Use Google Search → Start with "根据 Google 搜索结果（[current date]）：..."
- User: "你好" → Answer directly, no search needed → No prefix needed

CANVAS AWARENESS MODEL (CRITICAL):
The user's canvas may contain many items (documents, images, slides, webapps).
These are listed below as "Canvas items" - treat them as PASSIVE BACKGROUND KNOWLEDGE:

- **DO NOT** proactively read, analyze, or open ANY canvas items unless the user explicitly asks
- **DO NOT** use analyze_image or read_context_document on canvas items just because they exist
- You are aware of what exists on the canvas, but only access specific items when the user requests
- For general questions (greetings, casual chat, general knowledge): answer directly, IGNORE canvas items entirely
- For specific questions about a document/image: ONLY read/analyze the specific item the user mentions
- If the user says "look at the PDF" or "analyze the image", THEN use the appropriate tool on that specific item
- When uncertain whether to read an attachment, DON'T - just answer based on what you know

WHEN DOCUMENTS ARE SPECIFICALLY REQUESTED:
- Only read documents that the user explicitly mentions or asks about
- If the question is general and doesn't require document context, answer directly
- When you do read documents, summarize the relevant parts concisely

CRITICAL - EFFICIENCY AND ITERATION MANAGEMENT:
When analyzing multiple items (images, documents, files):
1. **Use sampling strategy**: Analyze 2-3 representative items first, NOT all items
2. **Identify patterns early**: Look for common themes/differences in your sample
3. **Spot-check if needed**: Only analyze 1-2 more items to verify patterns
4. **Provide answer based on sample**: Mention "Based on analysis of X representative items..."
5. **DO NOT analyze exhaustively**: Only analyze ALL items if user explicitly asks

Example GOOD approach (efficient):
  - Analyze images 1-3 → Identify design patterns → Spot-check image 5 → Answer
  - Total: ~5 tool calls

Example BAD approach (wasteful):
  - Analyze image 1 → Analyze 2 → Analyze 3 → ... → Analyze 10 → Answer
  - Total: 10+ tool calls before even starting to answer

When comparing items:
- Focus on HIGH-LEVEL patterns and KEY differences
- Use "good enough" principle - perfection not required
- After 15-20 tool calls, provide your answer with what you've learned
- You can always gather more details if user asks follow-up questions

IMAGE ANALYSIS - ONLY WHEN USER EXPLICITLY ASKS:
When user explicitly asks to analyze, describe, or look at a SPECIFIC image (keywords: "analyze", "describe", "what's in", "看看", "看图", "分析", "描述", "这张图"):
1. Use the analyze_image tool ONLY on the specific image the user mentions
2. Return the analysis result DIRECTLY - describe what you see in the image
3. Do NOT analyze images that the user did NOT ask about
4. Do NOT proactively analyze canvas images just because they exist in the attachment list

IMPORTANT GUIDELINES:
1. **Google Search**: USE IT for current info, real-time data, news, dates, prices, etc. Don't refuse search requests.
2. **Canvas items**: BACKGROUND CONTEXT ONLY - never read/analyze them unless user explicitly asks
3. When user asks about a specific canvas document/image, read ONLY that one
4. For general greetings ("你好"): answer directly with ZERO tool calls
5. For current information questions: USE Google Search - this is expected and correct
6. Be mindful of iteration budget - aim to answer within 20-30 tool calls maximum

Remember:
- Google Search = actively use when needed for real-time info
- Canvas items = passive context only, access when user asks"""


class ChatAgent:
    """Agent for intelligent chat responses with optional document access."""

    def __init__(self):
        self.llm_service = LLMService.get_instance(component="chat")
        self._agent_graph = None
        self._tools = []

    def _get_langchain_llm(self):
        """Get the appropriate LangChain LLM from the LLMService."""
        llm_service = LLMService.get_instance(component="chat")
        llm = llm_service.langchain_llm

        if llm is None:
            raise ValueError(
                f"LLM provider {settings.llm_provider} doesn't support LangChain integration"
            )

        return llm

    def create_tools(self, attachments: Optional[List] = None) -> List[BaseTool]:
        """Create tools for the chat agent.

        Args:
            attachments: Optional list of attachment objects

        Returns:
            List of LangChain tools
        """
        tools = []

        # Add context reading tool only if there are attachments available
        if attachments:
            # Extract paths from attachments
            document_s3_paths = []
            folder_paths = set()

            for att in attachments:
                # Handle both protobuf objects and dicts
                path = att.path if hasattr(att, "path") else att.get("path")
                if path:
                    document_s3_paths.append(path)
                    # Get the folder path (everything before the last /)
                    if "/" in path:
                        folder_path = "/".join(path.split("/")[:-1]) + "/"
                        folder_paths.add(folder_path)

            if document_s3_paths:
                tools.append(S3ContextReadTool(allowed_paths=document_s3_paths))

            # Add unified list tool for all folders
            if folder_paths:
                tools.append(S3ListTool(allowed_folder_paths=list(folder_paths)))

        # Add image analysis tool for analyzing image URLs
        tools.append(ImageAnalysisTool())

        # Add Google Search tool for real-time information
        tools.append(GoogleSearchTool())

        return tools

    async def create_agent_graph(
        self,
        attachments: Optional[List] = None,
        conversation_history: Optional[List[Dict]] = None,
    ):
        """Create an agent graph for chat responses using LangChain 1.0.

        Args:
            attachments: Optional list of attachment objects
            conversation_history: Optional conversation history for context

        Returns:
            Compiled agent graph
        """
        # Create tools
        tools = self.create_tools(attachments)
        self._tools = tools

        # Get the appropriate LangChain LLM
        llm = self._get_langchain_llm()
        logger.info(
            "Using LLM", llm_type=type(llm).__name__, provider=settings.llm_provider
        )

        # Build system prompt with current date and conversation history if available
        # Add current date to fix Gemini 3's time period confusion issue
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = f"IMPORTANT: Today's date is {current_date}. Use this date for all time-sensitive queries.\n\n{CHAT_SYSTEM_PROMPT}"

        if conversation_history:
            history_text = "\nPrevious conversation:\n"
            for msg in conversation_history[-20:]:  # Last 20 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                history_text += f"{role}: {content}\n"

            system_prompt = history_text + "\n" + system_prompt

        # Add attachment context, separating focused (selected) from background items
        if attachments:
            focused = []
            background = []
            for att in attachments:
                path = att.path if hasattr(att, "path") else att.get("path", "unknown")
                title = att.title if hasattr(att, "title") else att.get("title", "")
                att_type = (
                    att.type if hasattr(att, "type") else att.get("type", "document")
                )
                focus = att.focus if hasattr(att, "focus") else att.get("focus", False)
                entry = f'- ("{title}", "{att_type}", "{path}")' if title else f'- ("{att_type}", "{path}")'
                if focus:
                    focused.append(entry)
                else:
                    background.append(entry)

            doc_context = ""
            if focused:
                doc_context += "\nUser-selected items (FOCUS - proactively read/analyze these, the user is working with them):\n"
                doc_context += "\n".join(focused) + "\n"
            if background:
                doc_context += "\nOther canvas items (background awareness only - DO NOT read/analyze unless user explicitly asks):\n"
                doc_context += "\n".join(background) + "\n"
            if doc_context:
                system_prompt = system_prompt + doc_context

        # Create agent using LangChain 1.0 create_agent()
        logger.info("Creating agent graph", verbose=settings.debug)
        try:
            agent_graph = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(
                "Failed to create agent",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

        self._agent_graph = agent_graph
        return agent_graph

    @log_timing
    async def chat(
        self,
        user_prompt: str,
        attachments: Optional[List] = None,
        conversation_history: Optional[List[Dict]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Generate a chat response.

        Args:
            user_prompt: User's question or message
            attachments: Optional list of attachment objects
            conversation_history: Optional conversation history

        Returns:
            Chat response as a string
        """
        try:
            logger.info(
                "Starting chat response",
                request_id=request_id,
                has_attachments=bool(attachments),
                attachment_count=len(attachments) if attachments else 0,
            )

            # Create or recreate agent graph if needed
            if not self._agent_graph or self._tools != self.create_tools(
                attachments
            ):
                self._agent_graph = await self.create_agent_graph(
                    attachments, conversation_history
                )

            # Prepare the input message
            user_message = user_prompt

            # Add a hint about attachments if provided
            if attachments:
                user_message += "\n\n(Note: Attachments are available if you need them, but only read them if necessary to answer the question)"

            # Use shared utility for consistent streaming and logging
            result = await invoke_agent_with_streaming(
                agent_graph=self._agent_graph,
                user_input=user_message,
                logger=logger,
                request_id=request_id,
                agent_name="chat_agent",
            )

            response = result.get("output", "I apologize, but I couldn't generate a response.")

            return response

        except Exception as e:
            logger.error(
                "Chat response failed",
                operation="chat",
                has_attachments=bool(attachments),
                attachment_count=len(attachments) if attachments else 0,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return f"I encountered an error while processing your request: {str(e)}"

    @log_timing
    async def chat_stream(
        self,
        user_prompt: str,
        attachments: Optional[List] = None,
        conversation_history: Optional[List[Dict]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response.

        Args:
            user_prompt: User's question or message
            attachments: Optional list of attachment objects
            conversation_history: Optional conversation history

        Yields:
            Chunks of the chat response
        """
        try:
            logger.info(
                "Starting streaming chat response",
                request_id=request_id,
                has_attachments=bool(attachments),
                attachment_count=len(attachments) if attachments else 0,
            )

            # Create agent graph
            agent_graph = await self.create_agent_graph(
                attachments, conversation_history
            )

            # Prepare the input message
            user_message = user_prompt

            # Add a hint about attachments if provided
            if attachments:
                user_message += "\n\n(Note: Attachments are available if you need them, but only read them if necessary to answer the question)"

            # Use LangChain 1.0 streaming with graph.astream()
            logger.debug("Agent streaming started", method="astream")

            total_streamed = 0

            async for message, metadata in agent_graph.astream(
                {"messages": [HumanMessage(content=user_message)]},
                stream_mode="messages"
            ):
                # Stream AI message content chunks
                if isinstance(message, AIMessage) and message.content:
                    # Stream the content chunk
                    content_chunk = message.content
                    if isinstance(content_chunk, str) and content_chunk:
                        total_streamed += len(content_chunk)
                        yield content_chunk

        except Exception as e:
            logger.error(
                "Chat streaming failed",
                operation="chat_stream",
                has_attachments=bool(attachments),
                attachment_count=len(attachments) if attachments else 0,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            yield f"I encountered an error: {str(e)}"
