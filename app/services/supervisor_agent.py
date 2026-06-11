"""Intelligent supervisor agent that routes requests using LLM."""

import json
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

from app.core.logging import get_logger
from app.core.logging_utils import log_timing
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class RequestType(Enum):
    """Types of requests the supervisor can route to."""

    CHAT = "chat"
    GENERATION = "generation"


class SupervisorAgent:
    """Intelligent agent that routes requests using LLM."""

    def __init__(self):
        """Initialize the supervisor with LLM service."""
        # Initialize LLM service with router-specific model
        self.llm_service = LLMService.get_instance(component="router")

        # System prompt for routing decisions
        self.routing_system_prompt = """You are a request routing assistant. Analyze the user's prompt and determine if it's a Q&A request or a generation request.

When you see "Recent conversation:" followed by the current request, use the conversation history to understand context. For example:
- If the assistant asked clarifying questions about generation and the user responds, it's likely still a GENERATION request
- If discussing features/requirements for something to be built, it's likely GENERATION
- Simple follow-up questions about previous responses are usually CHAT

Q&A requests (CHAT) are when users:
- Ask questions about existing content
- Want explanations or analysis
- Seek information or clarification
- Request help understanding something
- Have general conversations or discussions
- Ask for ideas, suggestions, or brainstorming (e.g., "give me some ideas", "what do you think", "suggest me")
- Want to discuss or plan before actually generating (e.g., "I want to build X, what should I consider?")
- Seek advice or recommendations about what to create

Generation requests are when users:
- Give a direct command to create something NOW (not just discussing it)
- Ask to build, make, generate, or develop something immediately
- Request modifications to existing content that require generation
- Are responding to clarifying questions about what to generate
- Say things like "go ahead", "yes generate it", "create it now" after discussing requirements

IMPORTANT: If the user mentions wanting to generate/build/create something BUT is asking for ideas, suggestions, or discussion first, classify as CHAT. Only classify as GENERATION when they want immediate creation.

Examples:
- "I want to generate a webapp for my restaurant, give me some ideas" → CHAT (asking for ideas)
- "Generate a webapp for my restaurant" → GENERATION (direct command)
- "What kind of website would work best for my restaurant?" → CHAT (seeking advice)
- "Build me a restaurant website with online ordering" → GENERATION (direct command)

CRITICAL: Return ONLY valid JSON, no other text before or after.
Ensure all strings are properly quoted and escaped.
Do NOT include markdown formatting like ```json.

Return EXACTLY this structure:
{
  "request_type": "CHAT" or "GENERATION",
  "reasoning": "Brief explanation of your decision"
}"""

    @log_timing
    async def analyze_request(self, user_prompt: str) -> Tuple[RequestType, str]:
        """Analyze the user prompt and determine request type using LLM with regex fallback.

        Args:
            user_prompt: The user's input prompt

        Returns:
            Tuple of (RequestType, reasoning)
        """
        logger.debug("Analyzing user prompt", prompt_preview=user_prompt[:100])

        try:
            # Use LLM to analyze the request
            messages = LLMService.create_messages(
                system_prompt=self.routing_system_prompt, user_prompt=user_prompt
            )

            # Get LLM response using config from YAML
            response = await self.llm_service.generate(messages=messages)

            # Log raw response for debugging
            logger.debug(
                "Raw routing response",
                response_preview=response[:200] if response else "EMPTY RESPONSE",
            )

            # Parse JSON response - handle Gemini's markdown code blocks
            response_text = LLMService.extract_json_from_response(response)
            result = json.loads(response_text)

            # Validate and extract request type
            request_type_str = result.get("request_type", "").upper()
            reasoning = result.get("reasoning", "")

            if request_type_str == "GENERATION":
                logger.debug("Classified as generation request", reasoning=reasoning)
                return (RequestType.GENERATION, reasoning)
            elif request_type_str == "CHAT":
                logger.debug("Classified as chat request", reasoning=reasoning)
                return (RequestType.CHAT, reasoning)
            else:
                raise ValueError(f"Invalid request_type: {request_type_str}")

        except Exception as e:
            # Log the error and re-raise to let the caller handle it
            logger.error(
                "Request routing failed",
                operation="analyze_request",
                prompt_length=len(user_prompt),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Re-raise the exception so it bubbles up to the user
            raise e

    @log_timing
    async def plan_generation_tasks(
        self, user_prompt: str, conversation_context: str = "", attachments: List = None
    ) -> List[Dict[str, str]]:
        """Plan what artifacts to generate based on the prompt using LLM.

        Args:
            user_prompt: The user's generation request
            conversation_context: Previous conversation context
            attachments: List of attachment objects with path, type, title

        Returns:
            List of task dictionaries with 'title', 'type', and 'description' fields
        """
        # Check if user has image attachments (including ppt_png which are PPT slides rendered as images)
        has_image_attachment = False
        has_ppt_png_attachment = False
        if attachments:
            for att in attachments:
                att_type = att.get("type", "") if isinstance(att, dict) else getattr(att, "type", "")
                if att_type.startswith("image/"):
                    has_image_attachment = True
                elif att_type == "ppt_png":
                    has_ppt_png_attachment = True
                    has_image_attachment = True  # ppt_png is also an editable image

        # Build attachment context for the prompt
        attachment_info = ""
        if has_ppt_png_attachment:
            # PPT slides need special handling - user wants to EDIT existing slides, not generate new PPT
            attachment_info = "\n\nIMPORTANT: The user has attached PPT SLIDE IMAGES (ppt_png type). These are EXISTING slides from a PowerPoint. When the user wants to edit/modify/change these slides, use type 'image_edit' NOT 'ppt'. Only use 'ppt' when creating a completely NEW presentation from scratch."
        elif has_image_attachment:
            attachment_info = "\n\nIMPORTANT: The user has attached an IMAGE file. Consider whether they want to EDIT the attached image or GENERATE a new image."

        planning_system_prompt = f"""You are a task planning assistant. Analyze the user's generation request and plan what artifacts need to be created.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no text before or after
2. Do NOT use markdown formatting like ```json
3. Ensure all strings are properly quoted with double quotes
4. Escape any special characters in strings (\", \\, \n, etc.)
5. Return the raw JSON object starting with {{ and ending with }}

Available artifact types:
- webapp: For websites, web applications, landing pages, HTML content, interactive interfaces
- document: For markdown documents, reports, guides, PDFs, text content, documentation (文档), technical docs, READMEs
- image: For generating NEW images from scratch (text-to-image, when user has NO image attachment or wants something completely new)
- image_edit: For editing/modifying an EXISTING image (when user has attached an image AND wants to change/modify/edit it)
- ppt: For PowerPoint presentations, slides, slide decks, PPT files (幻灯片, 演示文稿, PPT)

TASK CLASSIFICATION RULES:
1. If user asks for "documentation", "docs", "文档", "guide", "report", "text" → type: "document"
2. If user asks for website, webapp, HTML, interface, dashboard → type: "webapp"
3. If user asks for "presentation", "slides", "PPT", "PowerPoint", "幻灯片", "演示文稿" → type: "ppt"

IMAGE VS IMAGE_EDIT RULES - CRITICAL:
- type: "image" → User wants to CREATE/GENERATE a brand new image from text description
- type: "image_edit" → User has attached an image AND wants to MODIFY/CHANGE/EDIT it

When to use "image_edit":
- User attached an image AND uses words like: change, modify, edit, remove, add, make it, turn it, convert, transform, adjust, fix, update, replace
- User attached an image AND uses Chinese words like: 改, 修改, 改变, 换成, 变成, 调整, 替换, 改为, 换为, 编辑, 删除, 添加, 转换
- Example: User attached car.jpg and says "change the color to red" → image_edit

When to use "image":
- User has NO image attachment and wants to generate something new
- User explicitly says "generate", "create new", "make a new image"
- Example: "Generate an image of a sunset" → image{attachment_info}

PPT SPECIAL RULES - CRITICAL FOR PRESENTATIONS:
1. For PPT/presentation requests: The number of slides is a PARAMETER, NOT a task count
2. ALWAYS create ONLY ONE task for PPT generation, regardless of how many slides are mentioned
3. Include the slide count in the description field as part of the generation parameters
4. Example: "3 slide PPT about anime" → ONE task with description mentioning "3 slides"
5. Example: "5-slide presentation" → ONE task with description mentioning "5 slides"
6. PPT generation creates ONE .pptx file containing multiple slides inside it

IMAGE_EDIT SPECIAL RULES - CRITICAL FOR BATCH IMAGE EDITING:
1. For image_edit requests: Create ONLY ONE task regardless of how many images are attached
2. The single task will process ALL attached images sequentially in order
3. Do NOT create separate tasks for each image - this causes ordering issues
4. Example: User has 3 images attached and says "change all to red" → ONE image_edit task
5. Example: User has 2 slides attached and says "fix the colors" → ONE image_edit task
6. The description should mention what edit to apply (will be applied to all attached images) or designated image to edit (E.G use image_2.png as reference to edit image_1.png).

QUANTITY RULES - EXTREMELY IMPORTANT (DOES NOT APPLY TO PPT OR IMAGE_EDIT):
1. Count the exact number requested: "two images" = 2 tasks, "three webapps" = 3 tasks
2. Look for number words: one, two, three, four, five, etc.
3. Look for digits: 1, 2, 3, 4, 5, etc.
4. If user says "multiple", "several", "a few" → default to 3 tasks
5. Each task MUST be separate - NEVER combine multiple items into one task

TASK SEPARATION:
1. Each distinct item MUST be a SEPARATE task in the array
2. Do NOT create one task that generates multiple files
3. Each task produces ONE artifact

The 'description' field should:
- Include ALL context needed for generation
- Be self-contained and specific
- Clearly specify what makes this variation unique

JSON FORMAT - FOLLOW EXACTLY:
{{
  "tasks": [
    {{
      "title": "Brief title max 5 words",
      "type": "webapp or document or image or image_edit or ppt",
      "description": "Complete prompt for generation agent"
    }}
  ]
}}

Examples:
Request: "Generate two images of cats" (no image attachment)
Response: {{"tasks":[{{"title":"Cat in window","type":"image","description":"Generate a realistic image of a domestic cat sitting in a sunny window"}},{{"title":"Cat with toy","type":"image","description":"Generate a realistic image of a playful cat with a toy mouse"}}]}}

Request: "Change the car color to red" (user attached car.jpg)
Response: {{"tasks":[{{"title":"Edit car color","type":"image_edit","description":"Change the car color to red"}}]}}

Request: "把车的颜色改成红色" (user attached an image)
Response: {{"tasks":[{{"title":"修改车颜色","type":"image_edit","description":"把车的颜色改成红色"}}]}}

Request: "Create documentation for my API"
Response: {{"tasks":[{{"title":"API documentation","type":"document","description":"Generate comprehensive markdown documentation for a REST API including endpoints, authentication, request/response examples, and error codes"}}]}}

Request: "Build three different landing pages"
Response: {{"tasks":[{{"title":"Modern landing page","type":"webapp","description":"Generate a modern minimalist landing page with hero section, features grid, and contact form"}},{{"title":"Corporate landing page","type":"webapp","description":"Generate a professional corporate landing page with navigation, services section, and testimonials"}},{{"title":"Startup landing page","type":"webapp","description":"Generate a vibrant startup landing page with animations, pricing table, and call-to-action buttons"}}]}}

Request: "What is anime? 3 slides. English. General."
Response: {{"tasks":[{{"title":"Anime Introduction PPT","type":"ppt","description":"Generate a PowerPoint presentation introducing anime with 3 slides in English using general template"}}]}}

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT."""

        try:
            # Include conversation context if available
            full_prompt = user_prompt
            if conversation_context:
                full_prompt = f"Previous conversation:\n{conversation_context}\n\nCurrent request: {user_prompt}"

            # Use LLM to plan tasks
            messages = LLMService.create_messages(
                system_prompt=planning_system_prompt, user_prompt=full_prompt
            )

            response = await self.llm_service.generate(messages=messages)

            # Log raw response for debugging
            logger.debug(
                "Raw LLM response for task planning",
                response_preview=response[:500] if response else "EMPTY RESPONSE",
            )

            # Parse JSON response - handle Gemini's markdown code blocks
            response_text = LLMService.extract_json_from_response(response)

            # Try to fix common JSON issues before parsing
            # Remove any trailing commas before closing brackets/braces
            response_text = re.sub(r",\s*([}\]])", r"\1", response_text)

            # Try to parse the JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # Log the specific JSON error for debugging
                logger.warning(
                    "JSON parsing failed",
                    error=str(json_err),
                    error_type=type(json_err).__name__,
                )
                logger.debug("Failed JSON content", content=response_text)

                # Try one more recovery attempt - extract just the tasks array
                tasks_match = re.search(
                    r'"tasks"\s*:\s*\[(.*?)\]', response_text, re.DOTALL
                )
                if tasks_match:
                    try:
                        # Wrap extracted content in proper JSON structure
                        tasks_json = '{"tasks": [' + tasks_match.group(1) + "]}"
                        result = json.loads(tasks_json)
                    except:
                        raise json_err
                else:
                    raise json_err

            tasks = result.get("tasks", [])

            # Validate tasks
            valid_types = {"webapp", "document", "image", "image_edit", "ppt"}
            validated_tasks = []

            for task in tasks:
                if isinstance(task, dict) and task.get("type") in valid_types:
                    validated_tasks.append(
                        {
                            "title": task.get("title", "Generated content"),
                            "type": task["type"],
                            "description": task.get("description", user_prompt),
                        }
                    )

            if validated_tasks:
                logger.debug("Tasks planned", task_count=len(validated_tasks))
                return validated_tasks
            else:
                raise ValueError("No valid tasks planned")

        except Exception as e:
            logger.error(
                "Task planning failed",
                operation="plan_generation_tasks",
                prompt_length=len(user_prompt),
                has_context=bool(conversation_context),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Re-raise the exception so it bubbles up to the user
            raise e
