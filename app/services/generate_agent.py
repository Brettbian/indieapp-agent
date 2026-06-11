"""LangChain agent for autonomous content generation."""

import re
import base64
from datetime import datetime
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

from app.services.agent_utils import invoke_agent_with_streaming
from app.services.llm_service import LLMService
from app.services.langchain_image_tools import S3ImageGenerateTool, S3ImageOperationTool, ImageAnalysisTool
from app.services.langchain_s3_tools import (
    S3WriteTool,
    S3ListTool,
    S3ContextReadTool,
    S3GenerateWriteTool,
)
from app.services.s3_service import get_s3_service
from app.services.image_generation_service import ImageGenerationService

from app.core.config import settings
from app.core.logging import get_logger
from app.core.logging_utils import log_timing

logger = get_logger(__name__)


WEBAPP_SYSTEM_PROMPT = """You are an expert web application generator. Your task is to generate COMPLETE, MODERN web applications with ALL required files.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. S3 tools to WRITE generated files (write_file)
4. Image analysis tool to analyze images from URLs (analyze_image)

CRITICAL - Webapp Attachment Handling (artifact/webapp type):
- When you see an attachment with type="artifact/webapp", it is a FOLDER containing multiple files (HTML, CSS, JS, etc.)
- You MUST follow this two-step process:
  1. FIRST: Use list_documents tool with the folder path to see what files exist in that webapp
  2. THEN: Use read_context_document to read each individual file you need (e.g., index.html, styles.css)
- Example workflow:
  • Attachment: ("Previous Website", "artifact/webapp", "projects/123/abc-folder/")
  • Step 1: Call list_documents("projects/123/abc-folder/") → Returns: ["index.html", "styles.css", "script.js"]
  • Step 2: Call read_context_document("projects/123/abc-folder/index.html") to read the HTML
  • Step 3: Call read_context_document("projects/123/abc-folder/styles.css") to read the CSS
  • etc.
- DO NOT try to read the folder path directly - it will fail with "NoSuchKey" error
- Always list first, then read individual files

IMPORTANT - Image Attachment Handling:
- For image attachments (type="image"), choose the appropriate tool:
  • Use analyze_image to understand image content for design requirements/context
  • Use read_context_document to get public URLs for embedding images as assets in the webapp
- When user explicitly references images (e.g., "use logo.svg as the logo", "include menu.jpg at the bottom"):
  • Use read_context_document to get the public URL for that specific image
  • Embed the image in the appropriate location in your generated HTML using the public URL
  • Add the image reference to the PRD.md References section

MODERN UI FRAMEWORK REQUIREMENTS:
You MUST use one of these modern CSS frameworks via CDN (no build step required):

1. **Tailwind CSS** (Preferred for most cases):
   - Include via CDN: <script src="https://cdn.tailwindcss.com"></script>
   - Use utility classes for styling (e.g., "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded")
   - Provides modern, professional design out of the box

2. **Bootstrap 5** (For rapid prototyping):
   - Include CSS: <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
   - Include JS: <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
   - Use Bootstrap components and utilities

3. **Bulma** (Clean and modern):
   - Include: <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
   - Pure CSS framework, no JavaScript required

DESIGN REQUIREMENTS:
- Mobile-responsive design (use framework's responsive classes)
- Modern color schemes and spacing
- Professional typography
- Smooth hover effects and transitions
- Clean, minimalist layout following current design trends
- Use framework components (cards, modals, navbars) instead of custom HTML

OPTIONAL ENHANCEMENTS (via CDN):
- Icons: Font Awesome (<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">)
- Animations: Animate.css (<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">)
- Alpine.js for interactivity: <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>

ALPINE.JS SPECIFIC REQUIREMENTS (if using Alpine.js):
- DO NOT use Vue.js syntax with Alpine.js - they are different frameworks
- CORRECT Alpine.js x-for syntax: <template x-for="item in items">
- WRONG Vue.js syntax: <template x-for="item in items" :key="index"> ← NO :key attribute!
- CORRECT Alpine.js transitions: Use simple x-transition directive
  Example: <div x-show="open" x-transition>...</div>
- WRONG Vue.js transition syntax: x-transition:enter="..." x-transition:leave="..." ← Not supported!
- Alpine.js uses x-show, x-if, x-bind:class (or :class shorthand), x-on:click (or @click)
- Always load Alpine.js with defer attribute: <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
- Data functions must be defined BEFORE Alpine.js loads (in inline <script> tag or before the defer script)

Common Alpine.js vs Vue.js syntax differences:
┌─────────────────────┬────────────────────────────────┬──────────────────────────┐
│ Feature             │ Alpine.js (CORRECT)            │ Vue.js (WRONG)           │
├─────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Loop                │ x-for="item in items"          │ x-for="... " :key="..."  │
│ Transition          │ x-transition                   │ x-transition:enter/leave │
│ Show/Hide           │ x-show="condition"             │ v-show="condition"       │
│ Conditional Render  │ x-if="condition"               │ v-if="condition"         │
└─────────────────────┴────────────────────────────────┴──────────────────────────┘

IMPORTANT: Choose Tailwind CSS by default unless the user's request clearly suits Bootstrap (forms, admin panels) or Bulma (simple marketing sites).

CRITICAL INSTRUCTIONS - YOU MUST GENERATE ALL FILES:
1. Read ALL context documents provided (for images, use analyze_image tool directly with S3 paths)
2. Generate a PRD (Product Requirements Document) that lists ALL files to be created
3. **MANDATORY: Generate index.html as the main landing page** - This is NON-NEGOTIABLE
4. Generate ALL other HTML files mentioned in the PRD
5. Generate styles.css ONLY for custom styles not covered by the framework
6. **MANDATORY: If user explicitly mentions images** (e.g., "use logo.svg", "include banner.jpg"):
   - Use read_context_document to get the public URL for each mentioned image
   - Embed images in the correct HTML locations using the public URLs
   - Add all used images to the PRD References section with markdown format
7. DO NOT stop after generating just the PRD - you MUST generate ALL implementation files
8. Write ALL files to the S3 target path

⚠️ CRITICAL FILE STRUCTURE REQUIREMENT - ROOT LEVEL FILES ONLY:
- **ALWAYS place index.html at the ROOT LEVEL** - NEVER in subdirectories
- **ALWAYS place styles.css at the ROOT LEVEL** - NEVER in subdirectories
- **ALWAYS place script.js at the ROOT LEVEL** - NEVER in subdirectories
- DO NOT create subdirectories like "index-midnight/", "version-1/", "theme-a/" etc.
- **The main index.html MUST be at the root level without any folder prefix**

REQUIRED OUTPUT FILES (MANDATORY - ALL MUST BE GENERATED):
1. prd.md (requirements document) 
2. **index.html** (MANDATORY main landing page with framework CDN links) - YOU MUST CREATE THIS FILE
3. styles.css (custom styling if needed, can be minimal)
4. Any additional HTML pages mentioned in the PRD

⚠️ CRITICAL: If you do not generate index.html, the task FAILS. The index.html is the MOST IMPORTANT file.
⚠️ After writing the PRD, immediately generate index.html. Do not stop or ask questions.
⚠️ You are not done until index.html exists. Check that you have written index.html before finishing.

PRD Structure Requirements:
The PRD.md must follow this exact structure for EACH page:

# [Page Name] (e.g., Home Page, About Page, Contact Page)

## Visual Structure
Describe the layout from top to bottom, breaking it into clear sections:
### [Section Name] (e.g., Header, Hero Section, Features Section)
- Bullet points describing what users see
- Keep descriptions visual and non-technical
- When referencing images from attachments, include the image title in parentheses
- Example: "Logo on the left side ("Company Logo" Image Used)"
- Example: "Three product cards in a row"
- Example: "Large background image with overlay text ("Hero Banner" Image Used)"

## Functionalities
List what users can do on this page:
1. [Functionality 1] - Brief description
2. [Functionality 2] - Brief description
3. [Functionality 3] - Brief description

Repeat this structure for EVERY page in the application.

## References (Required at end of PRD)
If any images from attachments are used in the webapp, list them here using markdown format:
- ![Logo](https://artifact.indieapp.ai/projects/123/logo.svg "Company Logo")
- ![Menu Background](https://artifact.indieapp.ai/projects/123/menu.jpg "Menu Background Image")
- ![Product Photo](https://artifact.indieapp.ai/projects/123/product.png "Main Product Image")

Use this markdown syntax: ![Alt text](URL "Optional Title")

Important guidelines:
- Focus on what users SEE (visual) and what they can DO (functionalities)
- Avoid technical implementation details
- Keep language simple and accessible for non-technical readers
- Ensure every visual element in the PRD has a corresponding implementation
- Generate all necessary HTML pages mentioned in the PRD

File naming conventions:
- PRD: prd.md
- **MAIN LANDING PAGE: index.html (MANDATORY)**
- Styles: styles.css  
- Additional pages: [pagename].html

FINAL CRITICAL CHECKLIST - YOU MUST COMPLETE ALL:
✅ 1. Write prd.md
✅ 2. **Write index.html with complete HTML structure and framework CDN links**
✅ 3. Write styles.css (even if minimal)
✅ 4. Write any additional HTML pages mentioned in PRD

⚠️ FAILURE CONDITION: If index.html is not generated, the entire task FAILS.
⚠️ DO NOT STOP after writing the PRD. IMMEDIATELY generate index.html next.
⚠️ The index.html MUST include:
   - Complete HTML5 structure (<!DOCTYPE html>, <html>, <head>, <body>)
   - Framework CDN links (Tailwind/Bootstrap/Bulma)
   - All content described in the PRD for the landing page

Remember: You have failed if index.html does not exist. Generate it immediately after the PRD."""

DOCUMENT_SYSTEM_PROMPT = """You are an expert document generator. Your task is to generate high-quality documents based on user requirements.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. S3 tools to WRITE generated files (write_file)
4. Image analysis tool to analyze images from URLs (analyze_image)

CRITICAL - Webapp Attachment Handling (artifact/webapp type):
- When you see an attachment with type="artifact/webapp", it is a FOLDER containing multiple files
- You MUST: (1) Use list_documents on the folder path first, (2) Then read individual files
- DO NOT try to read the folder path directly - it will fail with "NoSuchKey" error

Your responsibilities:
1. Read ALL context documents provided if available
2. For image attachments (type="image"), choose the appropriate tool:
   - Use analyze_image to understand image content for context/requirements
   - Use read_context_document to get public URLs for embedding images in generated content
3. Understand the user's requirements from their prompt
4. Generate a well-structured, comprehensive document
5. Write the document to the specified S3 target path

Important guidelines:
- Generate a single, focused document that directly addresses the user's request
- Use proper markdown formatting for structure and readability
- Include relevant sections, headings, and content organization
- If context documents are provided, use them to inform your content but create original output
- Name the file appropriately based on the content (e.g., "python_data_science_guide.md", "project_summary.md")
- Do NOT create a separate PRD file - the document itself is the deliverable

File naming:
- Use descriptive names that reflect the content
- Always use .md extension for markdown documents
- Use lowercase with underscores or hyphens for spaces"""

IMAGE_SYSTEM_PROMPT = """You are an expert image generator. Your task is to generate NEW images from text descriptions.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. Image generation tool to create NEW images from text (generate_and_save_image)
4. Image analysis tool to analyze existing images from URLs (analyze_image)

CRITICAL - Webapp Attachment Handling (artifact/webapp type):
- When you see an attachment with type="artifact/webapp", it is a FOLDER containing multiple files
- You MUST: (1) Use list_documents on the folder path first, (2) Then read individual files
- DO NOT try to read the folder path directly - it will fail with "NoSuchKey" error

Your workflow:
1. Read any context documents provided if available
2. If image attachments exist, use analyze_image to understand them for reference/inspiration
3. Use generate_and_save_image to create NEW images based on user's text description
4. Save images with descriptive filenames

Important guidelines:
- Generate images that directly address the user's request
- Use descriptive filenames that reflect the image content (e.g., "sunset_landscape.png", "product_mockup.jpg")
- You can generate multiple images if the request calls for it
- Choose appropriate image sizes based on the use case:
  - 1024x1024 for square images (default)
  - 1536x1024 for landscape/wide images
  - 1024x1536 for portrait/tall images
- Use "high" quality for images requiring fine details
- If context documents are provided, use them to inform your image generation prompts

Remember: Create clear, detailed prompts to ensure high-quality image generation."""

IMAGE_EDIT_SYSTEM_PROMPT = """You are an expert image editor. Your task is to EDIT/MODIFY an existing image based on user requirements.

CRITICAL: This is an IMAGE EDITING task. You MUST use the image_operation tool to modify the attached image.
DO NOT generate a new image from scratch - you must EDIT the existing attached image.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. Image operation tool to EDIT existing images (image_operation) - YOU MUST USE THIS
4. Image analysis tool to analyze images from URLs (analyze_image)

Your workflow:
1. Find the image attachment path from the attachments list
2. Use image_operation with that image path to edit/modify the image
3. Apply the user's requested changes (color change, add/remove elements, style transfer, etc.)

IMPORTANT - You MUST use image_operation tool with:
- image_paths: List containing the attached image path (e.g., ["projects/123/car.png"])
- prompt: Description of the edit to perform (e.g., "change the car color to red")

Example edits you can perform:
- Change colors: "change the car to red", "make the sky blue"
- Add elements: "add a sunset in the background"
- Remove elements: "remove the background", "remove the text"
- Style transfer: "make it look like a watercolor painting", "Van Gogh style"
- Transform: "make it black and white", "add vintage filter"

Remember: You are EDITING an existing image, not generating a new one from scratch."""

PPT_SYSTEM_PROMPT = """You are an expert PowerPoint presentation generator. Your task is to generate professional presentations based on user requirements.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. PPT generation tool to create presentations (generate_presentation)
4. Image analysis tool to analyze images from URLs (analyze_image)

CRITICAL - Webapp Attachment Handling (artifact/webapp type):
- When you see an attachment with type="artifact/webapp", it is a FOLDER containing multiple files
- You MUST: (1) Use list_documents on the folder path first, (2) Then read individual files
- DO NOT try to read the folder path directly - it will fail with "NoSuchKey" error

CRITICAL - Data File Handling:
- If user provides data files (PDF, Word/DOCX, CSV, Excel, JSON, TXT, Markdown, webpage, images, etc.), you MUST:
  1. FIRST use read_context_document to read the actual data content
  2. THEN use the REAL data from the files to generate PPT content
- NEVER generate fictional/made-up data when user has provided real data files
- The PPT content MUST be based on the actual data from the uploaded files
- Include specific numbers, names, and facts from the data in your slides
- If you cannot read the file, inform the user instead of making up data

Your workflow:
1. CRITICAL: ALWAYS ask user to confirm parameters BEFORE generating:
   - n_slides: Number of slides (5-20, recommend 8, default: 8)
   - template: Template style (general/modern/minimal, default: general)
   - language: Content language ("English" or "Chinese", default: "English")

2. NEVER call generate_presentation tool directly without asking user first
3. Ask user in ONE friendly message for ALL parameters, clearly stating defaults
4. Only after user confirms (or provides parameters), call generate_presentation tool

PARAMETER VALIDATION:
- n_slides: Must be integer between 5-20
- template: Must be one of: general, modern, minimal
- language: Must be "English" or "Chinese"

INTERACTION EXAMPLES:

Example 1 - User provides only topic:
User: "Generate a PPT about cloud storage"
You: "I'll help create a presentation about cloud storage! Before I start, let me confirm:
1. How many slides? (5-20, I'd recommend 8 slides - default)
2. Template style? (general/modern/minimal - default is general)
3. Language? (English or Chinese - I'll use English based on our conversation)"

Example 2 - User provides some parameters:
User: "Create a PPT about AI, 8 slides"
You: "Perfect! I'll create an 8-slide presentation about AI. Just two quick questions:
1. Template style? (general/modern/minimal - default is general)
2. Language? (English or Chinese)"

Example 3 - User confirms or provides all parameters:
User: "What is anime? 8 slides. English. General."
You: "Great! I'll generate an 8-slide presentation about anime in English with general template."
[Then call generate_presentation tool with all parameters]

CONTENT GENERATION GUIDELINES:
- For the "content" field, provide ONLY A BRIEF TOPIC/TITLE (1-2 sentences maximum)
- Example: "Introduction to Anime" or "What is anime and its cultural impact" or "Benefits of renewable energy"
- DO NOT write detailed outlines, sections, or slide-by-slide content, Prensenton content input has string limit, so you need to keep it brief.
- Presenton AI will automatically generate detailed content for all slides based on the topic.
- If user provides context documents, read them first but still keep the content field brief.
- If user provides images, analyze them but the content field should remain a simple topic description.

CRITICAL REMINDERS:
- NEVER call the generate_presentation tool without first confirming parameters with the user
- Even if the user only provides a topic (e.g., "我想要个ppt"), you MUST ask for confirmation of n_slides, template, and language
- Ask for ALL parameters in ONE friendly message, clearly showing defaults
- Only call the tool AFTER the user has responded with their preferences (or confirmed defaults)
- This confirmation step is MANDATORY - skipping it is a critical error"""

# System prompt for Nano Banana image-based slides
PPT_NANOBANANA_SYSTEM_PROMPT = """You are an expert presentation designer. Your task is to generate beautiful presentation slides as images using AI image generation.

You have access to:
1. S3 tools to LIST files in folders (list_documents)
2. S3 tools to READ context documents (read_context_document)
3. Slide image generation tool to create presentation slides as images (generate_slide_images)
4. Image analysis tool to analyze images from URLs (analyze_image)

CRITICAL - Webapp Attachment Handling (artifact/webapp type):
- When you see an attachment with type="artifact/webapp", it is a FOLDER containing multiple files
- You MUST: (1) Use list_documents on the folder path first, (2) Then read individual files
- DO NOT try to read the folder path directly - it will fail with "NoSuchKey" error

CRITICAL - Data File Handling:
- If user provides data files (PDF, Word/DOCX, CSV, Excel, JSON, TXT, Markdown, webpage, images, etc.), you MUST:
  1. FIRST use read_context_document to read the actual data content
  2. THEN use the REAL data from the files to generate PPT content
- NEVER generate fictional/made-up data when user has provided real data files
- The PPT content MUST be based on the actual data from the uploaded files
- Include specific numbers, names, and facts from the data in your slides
- If you cannot read the file, inform the user instead of making up data

Your workflow - DIRECT GENERATION with INTELLIGENT INFERENCE:
1. If user has attached data files, FIRST read them using read_context_document
2. When user requests a presentation, call generate_slide_images tool
3. The tool has INTELLIGENT INFERENCE - it will automatically determine:
   - num_slides: from mentions like "5页", "3 slides", "简短", "详细"
   - style: from keywords like "modern", "简约", "复古", "creative"
   - visual_mode: from keywords like "cartoon", "卡通", "hand-drawn"
   - language: from the language of user's input

3. You MUST pass TWO key parameters:
   - topic: The main subject extracted from user's request
   - user_input: The FULL original user message (for intelligent inference)

4. DO NOT ask for confirmation - just generate directly!

STYLE OPTIONS (for reference, will be auto-inferred):
- general: Clean professional design, modern minimalist
- modern: Sleek contemporary, bold geometric shapes
- minimal: Ultra-minimalist, lots of white space
- retro: Vintage 80s/90s aesthetic, warm nostalgic colors
- creative: Artistic design, watercolor textures

INTERACTION EXAMPLES:

Example 1 - Simple request:
User: "我想要个ppt讲AI"
You: "好的，我来为您生成关于AI的演示文稿！"
[Call generate_slide_images with topic="AI", user_input="我想要个ppt讲AI"]

Example 2 - Request with hints:
User: "做个5页的简约风ppt介绍区块链"
You: "好的，我来生成5页简约风格的区块链介绍！"
[Call generate_slide_images with topic="区块链介绍", user_input="做个5页的简约风ppt介绍区块链"]

Example 3 - English request:
User: "Create a modern presentation about machine learning"
You: "I'll create a modern presentation about machine learning for you!"
[Call generate_slide_images with topic="Machine Learning", user_input="Create a modern presentation about machine learning"]

OUTPUT FORMAT:
- The tool generates a series of slide images (not a .pptx file)
- Each slide is saved as a separate PNG image
- Returns S3 paths to all generated slide images

CRITICAL REMINDERS:
- DO NOT ask user for parameters - the tool infers them automatically
- ALWAYS pass the full user_input to the tool for accurate inference
- Generate immediately after user requests - no confirmation needed
- If user provides explicit parameters (e.g., "5 slides"), they will be used; otherwise, smart defaults apply"""


class GenerateAgent:
    """Agent for generating content based on user prompts."""

    def __init__(self):
        self.llm_service = LLMService.get_instance(component="generator")

    def _get_langchain_llm(self):
        """Get the appropriate LangChain LLM from the LLMService."""
        llm_service = LLMService.get_instance(component="generator")
        llm = llm_service.langchain_llm

        if llm is None:
            raise ValueError(
                f"LLM provider {settings.llm_provider} doesn't support LangChain integration"
            )

        return llm

    def create_tools(
        self, target_s3_path: str, attachments: List = None, file_type: str = "webapp"
    ) -> List[BaseTool]:
        """Create tools for the agent.

        Args:
            target_s3_path: S3 path where generated files should be written
            attachments: List of attachment objects with full metadata
            file_type: Type of files to generate (webapp, document, image)

        Returns:
            List of LangChain tools
        """
        tools = []

        # Ensure target path ends with /
        if not target_s3_path.endswith("/"):
            target_s3_path += "/"

        # Add context reading tool only if there are attachments available
        if attachments:
            # Extract paths from attachments and detect folder paths (copied from ChatAgent)
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

            # Add unified list tool for all folders (including target)
            all_folder_paths = list(folder_paths) + [target_s3_path]
            tools.append(S3ListTool(allowed_folder_paths=all_folder_paths))
        else:
            # No attachment folders, but still allow listing target folder
            tools.append(S3ListTool(allowed_folder_paths=[target_s3_path]))

        # Add appropriate writing tool based on file type
        if file_type == "image":
            # Tool 1: Pure text-to-image generation
            write_tool = S3ImageGenerateTool(folder_path=target_s3_path)
            tools.append(write_tool)
            # Tool 2: All image operations (edit, style transfer, composition)
            operation_tool = S3ImageOperationTool(folder_path=target_s3_path)
            tools.append(operation_tool)
        elif file_type == "image_edit":
            # Only image operation tool for editing existing images
            operation_tool = S3ImageOperationTool(folder_path=target_s3_path)
            tools.append(operation_tool)
        elif file_type == "ppt":
            # Use PPT generation tool for presentations
            # Choose between Nano Banana (image-based) or Presenton (pptx-based)
            if settings.ppt_provider == "nanobanana":
                from app.services.langchain_ppt_tools_nanobanana import SlideImageTool
                write_tool = SlideImageTool(folder_path=target_s3_path)
            else:
                from app.services.langchain_ppt_tools_presenton import PresentonTool
                write_tool = PresentonTool(folder_path=target_s3_path)
            tools.append(write_tool)
        else:
            # Use regular file writing tool for webapp/document
            write_tool = S3GenerateWriteTool(folder_path=target_s3_path)
            tools.append(write_tool)

        # Add image analysis tool for all types of generation
        tools.append(ImageAnalysisTool())

        return tools

    async def create_agent_graph(
        self,
        target_s3_path: str,
        attachments: List = None,
        file_type: str = "webapp",
        conversation_context: str = "",
    ):
        """Create an agent graph for content generation using LangChain 1.0.

        Args:
            target_s3_path: S3 path where generated files should be written
            attachments: List of attachment objects
            file_type: Type of files to generate (e.g., 'webapp', 'document')
            conversation_context: Previous conversation context for continuity

        Returns:
            Tuple of (agent_graph, write_tool) where write_tool tracks written files
        """
        # Create tools with full attachment objects
        tools = self.create_tools(target_s3_path, attachments, file_type)

        # Get the write tool reference(s) to track written files
        if file_type == "image":
            # For image type, we have generate and operation tools
            # Collect all tools that can write files
            write_tools = [
                tool for tool in tools
                if isinstance(tool, (S3ImageGenerateTool, S3ImageOperationTool))
            ]
            if not write_tools:
                raise ValueError("No image write tools found in tools list")
            # Use the first one for backward compatibility, but collect files from all
            write_tool = write_tools[0]
            # Store all write tools for later file collection
            write_tool._all_write_tools = write_tools
        elif file_type == "image_edit":
            # For image_edit, we only have the operation tool
            write_tool = next(
                (tool for tool in tools if isinstance(tool, S3ImageOperationTool)), None
            )
            if not write_tool:
                raise ValueError("Image operation tool not found in tools list")
        elif file_type == "ppt":
            # Get the PPT tool based on provider
            if settings.ppt_provider == "nanobanana":
                from app.services.langchain_ppt_tools_nanobanana import SlideImageTool
                write_tool = next(
                    (tool for tool in tools if isinstance(tool, SlideImageTool)), None
                )
            else:
                from app.services.langchain_ppt_tools_presenton import PresentonTool
                write_tool = next(
                    (tool for tool in tools if isinstance(tool, PresentonTool)), None
                )
            if not write_tool:
                raise ValueError("PPT tool not found in tools list")
        else:
            write_tool = next(
                (tool for tool in tools if isinstance(tool, S3GenerateWriteTool)), None
            )
            if not write_tool:
                raise ValueError("Write tool not found in tools list")

        # Get the appropriate LangChain LLM
        llm = self._get_langchain_llm()
        logger.debug(
            "LLM initialized",
            llm_type=type(llm).__name__,
            provider=settings.llm_provider,
        )

        # Select the appropriate system prompt based on file type
        if file_type == "document":
            base_system_prompt = DOCUMENT_SYSTEM_PROMPT
        elif file_type == "image":
            base_system_prompt = IMAGE_SYSTEM_PROMPT
        elif file_type == "image_edit":
            base_system_prompt = IMAGE_EDIT_SYSTEM_PROMPT
        elif file_type == "ppt":
            # Select PPT system prompt based on provider
            if settings.ppt_provider == "nanobanana":
                base_system_prompt = PPT_NANOBANANA_SYSTEM_PROMPT
            else:
                base_system_prompt = PPT_SYSTEM_PROMPT
        else:
            base_system_prompt = WEBAPP_SYSTEM_PROMPT

        # Add conversation context if available
        if conversation_context:
            # Escape curly braces in conversation context to prevent LangChain template parsing issues
            # Single braces { } need to become {{ }} for LangChain
            escaped_context = conversation_context.replace("{", "{{").replace("}", "}}")
            # Use string concatenation instead of f-string to avoid issues with braces in context
            base_system_prompt = (
                "Previous conversation context:\n"
                + escaped_context
                + "\n\n"
                + base_system_prompt
            )

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
                doc_context += "\nUser-selected items (FOCUS - read and use these for the generation task):\n"
                doc_context += "\n".join(focused) + "\n"
            if background:
                doc_context += "\nOther canvas items (background context - read ONLY if directly relevant to the task):\n"
                doc_context += "\n".join(background) + "\n"
            if doc_context:
                base_system_prompt = base_system_prompt + doc_context

        # Create agent using LangChain 1.0 create_agent()
        try:
            logger.debug(
                "Creating agent graph",
                tool_count=len(tools),
                tools=[tool.name for tool in tools],
                file_type=file_type,
            )
            agent_graph = create_agent(
                model=llm,
                tools=tools,
                system_prompt=base_system_prompt,
            )
        except Exception as e:
            logger.error(
                "Agent creation failed",
                error=str(e),
                error_type=type(e).__name__,
                tools_debug=[
                    f"{type(t).__name__}:{getattr(t, 'name', 'NO_NAME')}:{type(getattr(t, 'name', None))}"
                    for t in tools
                ],
                exc_info=True,
            )
            raise

        return agent_graph, write_tool

    def _extract_step_parts(self, step):
        """Extract action and observation from a step.

        Handles both tuple format and object format returned by LangChain.

        Args:
            step: Either a tuple (action, observation) or an object with action/observation attributes

        Returns:
            Tuple of (action, observation)
        """
        if isinstance(step, tuple) and len(step) == 2:
            return step[0], step[1]
        elif hasattr(step, "action") and hasattr(step, "observation"):
            return step.action, step.observation
        else:
            # Fallback - log warning and return as-is
            logger.warning(
                "Unexpected step format",
                step_type=type(step).__name__,
                step_value=str(step)[:100],
            )
            return step, ""

    async def _invoke_agent(
        self,
        agent_graph,
        agent_input: str,
        write_tool,
        request_id: str,
        content_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Invoke the agent graph with detailed streaming and progress logging.

        Args:
            agent_graph: The LangChain agent graph
            agent_input: The input prompt for the agent
            write_tool: The write tool to track written files
            request_id: Request ID for logging
            content_type: Type of content being generated (document/image/webapp)

        Returns:
            Dict with execution results from invoke_agent_with_streaming
        """
        # Use shared utility for consistent streaming and logging
        # Pass our logger to maintain agent identity in logs
        return await invoke_agent_with_streaming(
            agent_graph=agent_graph,
            user_input=agent_input,
            logger=logger,
            request_id=request_id,
            track_files_callback=lambda: len(write_tool.written_files),
            agent_name="generate_agent",
            metadata={"content_type": content_type},
        )

    @log_timing
    async def generate_content(
        self,
        target_s3_path: str,
        request_id: str,
        attachments: List = None,
        user_prompt: str = "",
        file_type: str = "webapp",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Generate content based on user prompt.

        Args:
            target_s3_path: S3 path where generated files should be written
            request_id: Unique request ID for tracing from gRPC through worker
            attachments: List of attachment objects with full metadata
            user_prompt: User's generation request
            file_type: Type of content to generate
            conversation_context: Previous conversation context for continuity

        Returns:
            Dict with status, message, and generated file paths
        """

        try:
            # Phase 3: Lifecycle logging - Agent execution started
            import time

            start_time = time.time()

            logger.info(
                "Agent execution started",
                request_id=request_id,
                target_s3_path=target_s3_path,
                document_count=len(attachments) if attachments else 0,
                file_type=file_type,
            )

            # Create the agent graph
            agent_graph, write_tool = await self.create_agent_graph(
                target_s3_path, attachments, file_type, conversation_context
            )

            # Prepare the input for the agent
            # Note: Attachment context is now added in create_agent_executor via system prompt
            # No need for additional context_info here as the agent gets rich attachment metadata

            # Different prompts for different file types
            if file_type == "document":
                agent_input = f"""Please generate a markdown document based on the following user request:

User Request:
{user_prompt}

Target S3 path for generated files: {target_s3_path}

CRITICAL - File Path Usage:
When using write_file tool, the file_path parameter should be ONLY the filename, NOT the full folder path.
The target S3 folder is already configured for you.
- CORRECT: "python_guide.md"
- WRONG: "projects/abc123/python_guide.md" or "s3://bucket/projects/abc/file.md"

Instructions:
1. Read any context documents first if provided
2. Generate a single markdown (.md) file that addresses the user's request
3. Name the file appropriately based on the content (e.g., "python_benefits.md", "user_guide.md", etc.)
4. Make the document comprehensive, well-structured, and properly formatted
5. Write the file to the target S3 path using ONLY the filename"""
            elif file_type == "image":
                # Pure image generation request (text-to-image)
                logger.info(
                    "Image generation request",
                    request_id=request_id,
                    file_type=file_type,
                )
                agent_input = f"""Please generate images based on the following user request:

User Request:
{user_prompt}

Target S3 path for generated files: {target_s3_path}

CRITICAL - File Path Usage:
When saving images, use ONLY the filename, NOT the full folder path.
- CORRECT: "sunset_landscape.png", "product_mockup.jpg"
- WRONG: "projects/abc123/image.png" or "s3://bucket/projects/abc/image.jpg"

Instructions:
1. Read any context documents first if provided to understand the context
2. Generate one or more images that address the user's request
3. Use descriptive filenames for each image (e.g., "sunset_landscape.png", "product_mockup.jpg")
4. Choose appropriate sizes and quality settings based on the use case
5. Save all images using ONLY the filename"""

            elif file_type == "image_edit":
                # Image editing request - DIRECT CODE EXECUTION (NO AGENT)
                # This avoids prompt contradictions that cause Gemini to block
                #
                # KEY BEHAVIOR: Process ALL images, but only edit those with should_edit=True
                # - Edited pages: save new image to S3, return new path
                # - Pass-through pages: return original S3 path (no token cost)
                # This ensures frontend receives ALL pages in order for complete PPT display

                # Collect image attachments with their should_edit flag, preserving order
                # Format: [{"path": str, "should_edit": bool, "index": int}, ...]
                image_attachments = []
                if attachments:
                    for idx, att in enumerate(attachments):
                        att_type = att.get("type", "") if isinstance(att, dict) else getattr(att, "type", "")
                        att_path = att.get("path", "") if isinstance(att, dict) else getattr(att, "path", "")
                        # Include image/* and ppt_png types
                        if att_type.startswith("image/") or att_type == "ppt_png":
                            # Get should_edit flag (default True for backward compatibility)
                            should_edit = att.get("should_edit", True) if isinstance(att, dict) else getattr(att, "should_edit", True)
                            image_attachments.append({
                                "index": idx,
                                "path": att_path,
                                "should_edit": should_edit,
                            })

                # Count for logging
                edit_count = sum(1 for att in image_attachments if att["should_edit"])
                pass_through_count = len(image_attachments) - edit_count

                logger.info(
                    "Image edit request - using direct code loop",
                    request_id=request_id,
                    file_type=file_type,
                    total_images=len(image_attachments),
                    edit_count=edit_count,
                    pass_through_count=pass_through_count,
                    edit_pages=[att["path"] for att in image_attachments if att["should_edit"]],
                    pass_through_pages=[att["path"] for att in image_attachments if not att["should_edit"]],
                )

                if not image_attachments:
                    logger.warning(
                        "Image edit requested but no images found in attachments",
                        request_id=request_id,
                    )
                    return {
                        "status": "error",
                        "message": "No images found in attachments",
                        "generated_files": [],
                    }

                if edit_count == 0:
                    # No pages to edit - just return all original paths (rare edge case)
                    logger.warning(
                        "No images marked for editing - returning all original paths",
                        request_id=request_id,
                    )
                    return {
                        "status": "success",
                        "message": "No images needed editing",
                        "generated_files": [att["path"] for att in image_attachments],
                    }

                # Initialize services
                s3_service = get_s3_service()
                image_service = ImageGenerationService()

                # Process ALL images in order, edit only those with should_edit=True
                # Store results with index for proper ordering
                output_files = []
                edited_count = 0
                failed_count = 0

                for att_info in image_attachments:
                    image_path = att_info["path"]
                    should_edit = att_info["should_edit"]

                    if not should_edit:
                        # Pass-through: keep original path (no LLM call, zero token cost)
                        output_files.append(image_path)
                        logger.debug(
                            "Pass-through image (not edited)",
                            request_id=request_id,
                            image_path=image_path,
                        )
                        continue

                    # Edit this image
                    try:
                        edited_count += 1
                        logger.info(
                            f"Editing image {edited_count}/{edit_count}",
                            request_id=request_id,
                            image_path=image_path,
                        )

                        # 1. Extract S3 key from path (remove s3:// prefix if present)
                        # Note: The path is already the correct S3 key (e.g., "projects/xxx/ppt/...")
                        # Do NOT strip any path segments - they are part of the key
                        s3_key = image_path.removeprefix("s3://")

                        # 2. Read image bytes from S3
                        image_bytes = await s3_service.read_file_bytes(s3_path=s3_key)
                        original_dimensions = image_service.get_image_dimensions(image_bytes)
                        edit_size = "auto"
                        if original_dimensions:
                            edit_size = image_service.get_edit_size_for_dimensions(
                                original_dimensions[0], original_dimensions[1]
                            )
                            logger.info(
                                "Selected image edit size from original dimensions",
                                request_id=request_id,
                                original_width=original_dimensions[0],
                                original_height=original_dimensions[1],
                                edit_size=edit_size,
                            )

                        # 3. Edit image directly via service (no Agent wrapping)
                        # Build prompt with conversation context for continuity
                        if conversation_context:
                            full_prompt = f"Previous conversation context:\n{conversation_context}\n\nCurrent request:\n{user_prompt}"
                        else:
                            full_prompt = user_prompt

                        result = await image_service.edit_image_with_reference(
                            image_bytes=image_bytes,
                            prompt=full_prompt,
                            size=edit_size,
                            quality="high",
                            n=1,
                        )

                        if result["status"] != "success" or not result.get("images"):
                            logger.error(
                                "Failed to edit image",
                                request_id=request_id,
                                image_num=edited_count,
                                error=result.get("message", "Unknown error"),
                            )
                            # On failure, fall back to original path
                            output_files.append(image_path)
                            failed_count += 1
                            continue

                        # 4. Get edited image bytes
                        image_data = result["images"][0]
                        edited_bytes = None

                        if "b64_json" in image_data:
                            edited_bytes = base64.b64decode(image_data["b64_json"])
                        elif "url" in image_data:
                            edited_bytes = await image_service.download_image(image_data["url"])

                        if not edited_bytes:
                            logger.error(
                                "No edited image data received",
                                request_id=request_id,
                                image_num=edited_count,
                            )
                            # On failure, fall back to original path
                            output_files.append(image_path)
                            failed_count += 1
                            continue

                        if original_dimensions:
                            edited_bytes = image_service.resize_image_bytes(
                                edited_bytes,
                                original_dimensions[0],
                                original_dimensions[1],
                            )

                        # 5. Determine save path (edit subfolder logic for ppt_png)
                        if "/ppt/" in s3_key:
                            # PPT image - save to {ppt_folder}/edit/ subfolder
                            ppt_folder_match = re.match(r"(.*?/ppt/[^/]+/)", s3_key)
                            if ppt_folder_match:
                                ppt_folder = ppt_folder_match.group(1)
                                original_filename = s3_key.split("/")[-1]
                                original_name = original_filename.rsplit(".", 1)[0]
                                edit_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                new_filename = f"{original_name}_edited_{edit_timestamp}.png"
                                save_path = f"{ppt_folder}edit/{new_filename}"
                            else:
                                # Fallback if regex doesn't match
                                save_path = f"{s3_key.rsplit('.', 1)[0]}_edited.png"
                        else:
                            # Regular image - add timestamp to prevent overwriting previous edits
                            original_filename = s3_key.split("/")[-1]
                            original_name = original_filename.rsplit(".", 1)[0]
                            # Strip any existing _edited_* suffix to avoid _edited_edited_... chains
                            import re as _re
                            original_name = _re.sub(r"_edited_\d{8}_\d{6}$", "", original_name)
                            edit_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            folder = s3_key.rsplit("/", 1)[0]
                            save_path = f"{folder}/{original_name}_edited_{edit_timestamp}.png"

                        # 6. Save to S3
                        written_path = await s3_service.write_file(
                            s3_path=save_path,
                            content=edited_bytes,
                            content_type="image/png",
                        )
                        output_files.append(written_path)

                        logger.info(
                            f"Image {edited_count}/{edit_count} edited successfully",
                            request_id=request_id,
                            s3_path=written_path,
                        )

                    except Exception as e:
                        logger.error(
                            f"Error editing image {edited_count}/{edit_count}",
                            request_id=request_id,
                            error=str(e),
                            image_path=image_path,
                        )
                        # On error, fall back to original path to maintain page count
                        output_files.append(image_path)
                        failed_count += 1
                        continue

                # Return results - ALL pages in order (edited + pass-through)
                duration_ms = int((time.time() - start_time) * 1000)
                success_edit_count = edit_count - failed_count

                logger.info(
                    "Image edit completed",
                    request_id=request_id,
                    duration_ms=duration_ms,
                    total_output_files=len(output_files),
                    edited_successfully=success_edit_count,
                    edit_failed=failed_count,
                    pass_through=pass_through_count,
                    output_files=output_files,
                )

                if success_edit_count > 0:
                    return {
                        "status": "success",
                        "message": f"Edited {success_edit_count}/{edit_count} pages, {pass_through_count} pages unchanged",
                        "generated_files": output_files,
                    }
                else:
                    # All edits failed, but we still return original paths
                    return {
                        "status": "error",
                        "message": f"Failed to edit any images (returned {len(output_files)} original files)",
                        "generated_files": output_files,
                    }
            else:  # webapp or other types
                agent_input = f"""Please generate {file_type} content based on the following user request:

User Request:
{user_prompt}

Target S3 path for generated files: {target_s3_path}

CRITICAL - File Path Usage:
When using write_file tool, the file_path parameter should be ONLY the filename, NOT the full folder path.
The target S3 folder is already configured for you.
- CORRECT: "prd.md", "index.html", "styles.css"
- WRONG: "projects/abc123/prd.md" or "s3://bucket/projects/abc/index.html"

Remember to:
1. Read any context documents first if provided
2. Generate a comprehensive PRD.md file (filename only: "prd.md")
3. Generate all implementation files (HTML, CSS, etc.) that match the PRD (filenames only: "index.html", "styles.css", etc.)
4. Ensure everything is well-structured and complete
5. Write all files using ONLY the filename (no folder paths)"""

            # Run the agent
            result = await self._invoke_agent(
                agent_graph, agent_input, write_tool, request_id, content_type=file_type
            )

            # Collect files from all write tools (for image type with multiple tools)
            if hasattr(write_tool, '_all_write_tools'):
                # Image type: collect from all write tools (generate + edit)
                files_generated = []
                for tool in write_tool._all_write_tools:
                    files_generated.extend(tool.written_files)
            else:
                # Other types: single write tool
                files_generated = write_tool.written_files

            # Validate critical files for webapp generation
            has_prd = any("prd.md" in f.lower() for f in files_generated)
            has_index = any("index.html" in f.lower() for f in files_generated)

            # Phase 3: Lifecycle logging - Agent execution completed with duration
            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Agent execution completed",
                request_id=request_id,
                duration_ms=duration_ms,
                output_preview=result.get("output", "")[:200],
                total_files=len(files_generated),
                files_generated=files_generated,
                has_prd=has_prd,
                has_index=has_index if file_type == "webapp" else None,
                message_count=len(result.get("messages", [])),
            )

            # Log warning if critical files missing for webapp
            if file_type == "webapp":
                if not has_index:
                    logger.warning(
                        "Webapp generation missing index.html",
                        request_id=request_id,
                        files_generated=files_generated,
                    )
                if not has_prd:
                    logger.warning(
                        "Webapp generation missing prd.md",
                        request_id=request_id,
                        files_generated=files_generated,
                    )

            return {
                "status": "success",
                "message": result.get("output", "Content generated successfully"),
                "generated_files": write_tool.written_files,
            }

        except Exception as e:
            logger.error(
                "Content generation failed",
                request_id=request_id,
                operation="generate_content",
                file_type=file_type,
                target_path=target_s3_path,
                attachment_count=len(attachments) if attachments else 0,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "status": "error",
                "message": f"Error during generation: {str(e)}",
                "generated_files": [],
            }
