"""gRPC service implementation for AI backend."""

import asyncio
import uuid
from typing import AsyncIterator, Dict, List, Any
from concurrent import futures

import grpc
from grpc import aio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Import generated proto files (will be created by generate_proto.py)
from app.proto import ai_service_pb2
from app.proto import ai_service_pb2_grpc

from app.core.config import settings
from app.core.logging import get_logger
from app.services.s3_service import get_s3_service
from app.services.tasks import generate_content, edit_content
from app.services.supervisor_agent import RequestType
from app.services.chat_memory import get_chat_memory_service
from app.services.router_graph import create_router_graph

logger = get_logger(__name__)


class AiServicer(ai_service_pb2_grpc.AiServiceServicer):
    """Implementation of the AI gRPC service."""

    def __init__(self):
        """Initialize the service."""
        self.s3_service = get_s3_service()
        self.chat_memory = get_chat_memory_service()  # Deprecated: kept for backward compatibility
        self.task_callbacks = {}

        # Router graph with checkpointer (initialized lazily)
        self._router_graph = None
        self._checkpointer = None
        self._checkpointer_cm = None  # Context manager for checkpointer
        self._init_lock = asyncio.Lock()
        self._initialized = False
        # Issue 1 fix: per-thread_id locks to prevent concurrent graph
        # invocations on the same conversation (which could cause
        # double-compaction or checkpoint overwrites).
        self._thread_locks: Dict[str, asyncio.Lock] = {}
        self._thread_locks_lock = asyncio.Lock()  # Protects _thread_locks dict itself

        logger.info("AI gRPC service initialized")

    async def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create a per-thread_id asyncio.Lock.

        Prevents concurrent router_graph.ainvoke() calls on the same
        thread_id, which would race on checkpoint read/write.
        """
        if thread_id not in self._thread_locks:
            async with self._thread_locks_lock:
                # Double-check after acquiring meta-lock
                if thread_id not in self._thread_locks:
                    self._thread_locks[thread_id] = asyncio.Lock()
        return self._thread_locks[thread_id]

    async def _ensure_initialized(self):
        """Initialize async components (checkpointer, router graph) on first use."""
        if self._initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            # Initialize checkpointer from database URL
            # from_conn_string returns an async context manager, so we enter it
            self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                settings.database_url
            )
            self._checkpointer = await self._checkpointer_cm.__aenter__()
            await self._checkpointer.setup()  # Create tables if needed

            # Ensure compaction_archive table exists (Issue #70)
            # Archive is a safety net — if table creation fails, continue without it.
            try:
                import psycopg
                async with await psycopg.AsyncConnection.connect(
                    settings.database_url
                ) as conn:
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS compaction_archive (
                            id SERIAL PRIMARY KEY,
                            thread_id VARCHAR(255) NOT NULL,
                            archived_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            messages JSONB NOT NULL,
                            request_id VARCHAR(255),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    """)
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_compaction_archive_thread
                        ON compaction_archive(thread_id, archived_at)
                    """)
                    await conn.commit()
            except Exception as e:
                logger.warning(
                    "Failed to create compaction_archive table (non-fatal)",
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Create router graph with checkpointer
            self._router_graph = create_router_graph(checkpointer=self._checkpointer)

            self._initialized = True
            logger.info("Router graph with checkpointer initialized")

    async def Chat(
        self,
        request: ai_service_pb2.ChatRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[ai_service_pb2.ChatResponse]:
        """Handle unified chat requests - both Q&A and generation.

        Uses RouterGraph for classification, validation, and dispatching.
        Messages are automatically persisted via the checkpointer.
        """
        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())

        # Bind request_id to logger for this request
        request_logger = logger.bind(
            request_id=request_id, project_id=request.project_id
        )

        request_logger.info(
            "Chat request received",
            attachment_count=len(request.attachments) if request.attachments else 0,
            prompt_length=len(request.user_prompt) if request.user_prompt else 0,
        )

        try:
            # Initialize router graph and checkpointer on first use
            await self._ensure_initialized()

            # CONFIG: Immutable input (thread lookup + per-request data)
            config = {
                "configurable": {
                    "thread_id": request.project_id,  # Key for checkpointer
                    "request_id": request_id,  # For tracing
                    "attachments": [
                        {"path": att.path, "type": att.type, "title": att.title, "focus": att.focus}
                        for att in request.attachments
                    ]
                    if request.attachments
                    else [],
                }
            }

            # STATE: New message + reset inter-node fields
            # Previous messages are AUTO-LOADED from checkpointer and merged!
            input_state = {
                "messages": [HumanMessage(content=request.user_prompt)],
                # Reset all inter-node communication fields:
                "request_type": None,
                "classification_reasoning": "",
                "is_context_sufficient": True,
                "clarifying_questions": "",
                "planned_tasks": [],
                "celery_task_ids": [],  # Only store task IDs (serializable)
                "task_map": {},
                "task_type_map": {},
                "error": None,
            }

            # Issue 1 fix: serialize graph invocations per thread_id
            thread_lock = await self._get_thread_lock(request.project_id)
            async with thread_lock:
                final_state = await self._router_graph.ainvoke(input_state, config)

            # Stream response (outside the lock - streaming doesn't modify checkpoint)
            if final_state["request_type"] == RequestType.CHAT.value:
                # Chat request: stream the AI response
                async for response in self._stream_chat_response(
                    final_state, request_logger
                ):
                    yield response
            else:
                # Generation request: stream intro + monitor tasks
                async for response in self._stream_generation_response(
                    final_state, request, request_logger
                ):
                    yield response

        except Exception as e:
            request_logger.error(
                "Chat request failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

            # Stream error message to user instead of aborting
            error_msg = (
                f"I encountered an error while processing your request: {str(e)}"
            )

            # Check if it's a content filter error and provide helpful message
            if (
                "content_filter" in str(e).lower()
                or "content management policy" in str(e).lower()
            ):
                error_msg = "I'm unable to process this request due to content policy restrictions. Please rephrase your request or remove any potentially sensitive content from the conversation."
            elif "400" in str(e) and "bad request" in str(e).lower():
                error_msg = "I encountered a request format issue. This might be due to content restrictions or technical limitations. Please try rephrasing your request."

            yield ai_service_pb2.ChatResponse(content_chunk=error_msg)

    async def _stream_chat_response(
        self,
        final_state: dict,
        request_logger,
    ) -> AsyncIterator[ai_service_pb2.ChatResponse]:
        """Stream the chat response from router graph final state.

        Args:
            final_state: Final state from router graph with AI message
            request_logger: Logger with request context
        """
        # Get the AI response (last message in state)
        ai_message = final_state["messages"][-1]
        content = ai_message.content
        # Handle multimodal content blocks (Gemini 3 returns list format)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            response_content = "".join(text_parts)
        else:
            response_content = content if isinstance(content, str) else str(content)

        # Stream in chunks
        buffer = ""
        chunk_size = settings.streaming_chunk_size

        for i in range(0, len(response_content), chunk_size):
            chunk = response_content[i : i + chunk_size]
            yield ai_service_pb2.ChatResponse(content_chunk=chunk)

        request_logger.info(
            "Chat Q&A request completed",
            response_length=len(response_content),
        )

    async def _stream_generation_response(
        self,
        final_state: dict,
        request: ai_service_pb2.ChatRequest,
        request_logger,
    ) -> AsyncIterator[ai_service_pb2.ChatResponse]:
        """Stream generation response with intro message and task monitoring.

        Args:
            final_state: Final state from router graph with tasks
            request: Original chat request
            request_logger: Logger with request context
        """
        # Get the intro message (last message in state)
        ai_message = final_state["messages"][-1]
        content = ai_message.content
        # Handle multimodal content blocks (Gemini 3 returns list format)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            intro_message = "".join(text_parts)
        else:
            intro_message = content if isinstance(content, str) else str(content)

        # Stream the intro message in chunks
        chunk_size = settings.streaming_chunk_size
        for i in range(0, len(intro_message), chunk_size):
            yield ai_service_pb2.ChatResponse(
                content_chunk=intro_message[i : i + chunk_size]
            )

        # Get task data from final state
        task_map = final_state.get("task_map", {})
        task_type_map = final_state.get("task_type_map", {})
        task_return_type_map = final_state.get("task_return_type_map", {})
        celery_task_ids = final_state.get("celery_task_ids", [])

        # If no tasks (e.g., clarifying questions returned), we're done
        if not celery_task_ids:
            request_logger.info("No generation tasks to monitor")
            return

        # Send PENDING status for each task
        for task_id, title in task_map.items():
            # Use task_return_type_map for file_type (preserves ppt_png for edited images)
            # Fall back to task_type_map if not available
            file_type = task_return_type_map.get(task_id) or task_type_map.get(task_id, "webapp")
            # For new PPT generation, use ppt_png for nanobanana provider
            if file_type == "ppt" and settings.ppt_provider == "nanobanana":
                file_type = "ppt_png"

            yield ai_service_pb2.ChatResponse(
                task_update=ai_service_pb2.TaskUpdateResponse(
                    task_id=task_id,
                    title=title,
                    status=ai_service_pb2.TaskStatus.PENDING,
                    file_type=file_type,
                )
            )

        # Monitor tasks and send updates
        # Reconstruct AsyncResult objects from task IDs
        pending_tasks = set(celery_task_ids)
        task_states = {task_id: "PENDING" for task_id in pending_tasks}

        while pending_tasks:
            for task_id in list(celery_task_ids):
                if task_id not in pending_tasks:
                    continue

                # Reconstruct AsyncResult from task_id
                celery_task = generate_content.AsyncResult(task_id)

                # Check if task is ready
                if celery_task.ready():
                    pending_tasks.remove(task_id)

                    if celery_task.successful():
                        result = celery_task.result
                        output_paths = result.get("output_s3_paths", [])

                        # Check if any files were actually generated
                        if not output_paths:
                            yield ai_service_pb2.ChatResponse(
                                task_update=ai_service_pb2.TaskUpdateResponse(
                                    task_id=task_id,
                                    title=task_map[task_id],
                                    status=ai_service_pb2.TaskStatus.FAILED,
                                    error_message="No files were generated. The generation process failed to create any output.",
                                )
                            )
                        else:
                            yield ai_service_pb2.ChatResponse(
                                task_update=ai_service_pb2.TaskUpdateResponse(
                                    task_id=task_id,
                                    title=task_map[task_id],
                                    status=ai_service_pb2.TaskStatus.COMPLETED,
                                    output_s3_paths=output_paths,
                                )
                            )
                    else:
                        error = str(celery_task.info)
                        yield ai_service_pb2.ChatResponse(
                            task_update=ai_service_pb2.TaskUpdateResponse(
                                task_id=task_id,
                                title=task_map[task_id],
                                status=ai_service_pb2.TaskStatus.FAILED,
                                error_message=error,
                            )
                        )

                elif (
                    celery_task.state == "STARTED" and task_states[task_id] != "STARTED"
                ):
                    # Send IN_PROGRESS update once
                    task_states[task_id] = "STARTED"
                    yield ai_service_pb2.ChatResponse(
                        task_update=ai_service_pb2.TaskUpdateResponse(
                            task_id=task_id,
                            title=task_map[task_id],
                            status=ai_service_pb2.TaskStatus.IN_PROGRESS,
                        )
                    )

            # Small delay to avoid busy waiting
            if pending_tasks:
                await asyncio.sleep(1.0)

        request_logger.info(
            "Generation request completed",
            tasks_count=len(celery_task_ids),
        )

        # --- Issue 2 fix: inject task results into conversation history ---
        # After all tasks complete, write a SystemMessage summarizing what was
        # generated. This becomes part of the conversation history so the user
        # can reference generated artifacts in future messages.
        await self._inject_task_results(
            project_id=request.project_id,
            task_map=task_map,
            celery_task_ids=celery_task_ids,
            request_logger=request_logger,
        )

    async def _inject_task_results(
        self,
        project_id: str,
        task_map: Dict[str, str],
        celery_task_ids: List[str],
        request_logger,
    ) -> None:
        """Inject task completion results into the conversation checkpoint.

        After Celery tasks complete, this appends a SystemMessage to the
        conversation history summarizing what was generated and the S3 paths.

        Issue 2 fix: replaces the broken direct-checkpoint-manipulation approach.
        """
        try:
            # Build result summary
            result_parts = []
            for task_id in celery_task_ids:
                celery_task = generate_content.AsyncResult(task_id)
                title = task_map.get(task_id, "Unknown")
                if celery_task.successful():
                    result = celery_task.result
                    paths = result.get("output_s3_paths", [])
                    result_parts.append(f"- {title}: {', '.join(paths)}")
                else:
                    result_parts.append(f"- {title}: FAILED")

            if not result_parts:
                return

            result_text = (
                "[Task Completed] Generated content:\n"
                + "\n".join(result_parts)
            )

            await self._append_system_message(
                project_id=project_id,
                message=SystemMessage(
                    content=result_text,
                    id=f"task-result-{uuid.uuid4()}",
                ),
            )

            request_logger.info(
                "Task results injected into conversation",
                project_id=project_id,
                result_count=len(result_parts),
            )

        except Exception as e:
            request_logger.warning(
                "Failed to inject task results into conversation",
                project_id=project_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _append_system_message(
        self,
        project_id: str,
        message: SystemMessage,
    ) -> None:
        """Append a SystemMessage to a conversation's checkpoint.

        Uses the checkpointer's public API:
        1. aget_tuple() to read current checkpoint
        2. Append message to 'messages' channel
        3. aput() with correct signature

        Protected by per-thread_id lock to prevent races with concurrent Chat() calls.
        """
        thread_lock = await self._get_thread_lock(project_id)
        async with thread_lock:
            config = {"configurable": {"thread_id": project_id}}

            # Read current checkpoint
            checkpoint_tuple = await self._checkpointer.aget_tuple(config)
            if checkpoint_tuple is None:
                logger.warning(
                    "No checkpoint found for thread, skipping message injection",
                    thread_id=project_id,
                )
                return

            checkpoint = checkpoint_tuple.checkpoint
            metadata = checkpoint_tuple.metadata or {}

            # Get current messages from channel_values
            channel_values = checkpoint.get("channel_values", {})
            current_messages = channel_values.get("messages", [])

            # Append the new message
            current_messages.append(message)
            channel_values["messages"] = current_messages
            checkpoint["channel_values"] = channel_values

            # channel_versions tracks which channels have been updated
            channel_versions = checkpoint.get("channel_versions", {})

            # Write back using aput with the correct 4-argument signature
            await self._checkpointer.aput(
                config=checkpoint_tuple.config,
                checkpoint=checkpoint,
                metadata={**metadata, "source": "task_result_injection"},
                new_versions=channel_versions,
            )

    async def Edit(
        self,
        request: ai_service_pb2.EditRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[ai_service_pb2.TaskUpdateResponse]:
        """Handle asynchronous content editing requests."""
        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())
        request_logger = logger.bind(request_id=request_id)

        request_logger.info(
            "Edit request received",
            s3_folder_path=request.s3_folder_path,
            prompt_length=len(request.prompt),
        )

        try:
            task_id = str(uuid.uuid4())

            # Send initial PENDING status
            yield ai_service_pb2.TaskUpdateResponse(
                task_id=task_id,
                title="Edit content",
                status=ai_service_pb2.TaskStatus.PENDING,
            )

            # Create callback for this task
            callback_queue = asyncio.Queue()
            self.task_callbacks[task_id] = callback_queue

            # Submit task to Celery
            task = edit_content.apply_async(
                args=[
                    task_id,
                    request.s3_folder_path,
                    request.prompt,
                ],
                task_id=task_id,
            )

            # Monitor task
            sent_in_progress = False
            while not task.ready():
                if task.state == "STARTED" and not sent_in_progress:
                    sent_in_progress = True
                    yield ai_service_pb2.TaskUpdateResponse(
                        task_id=task_id,
                        title="Edit content",
                        status=ai_service_pb2.TaskStatus.IN_PROGRESS,
                    )

                await asyncio.sleep(1.0)  # Increased delay to reduce polling

            # Send final status
            if task.successful():
                result = task.result
                yield ai_service_pb2.TaskUpdateResponse(
                    task_id=task_id,
                    title="Edit content",
                    status=ai_service_pb2.TaskStatus.COMPLETED,
                    output_s3_paths=result.get("output_s3_paths", []),
                )
            else:
                error = str(task.info)
                yield ai_service_pb2.TaskUpdateResponse(
                    task_id=task_id,
                    title="Edit content",
                    status=ai_service_pb2.TaskStatus.FAILED,
                    error_message=error,
                )

            request_logger.info(
                "Edit request completed",
                s3_folder_path=request.s3_folder_path,
                task_id=task_id,
            )

        except Exception as e:
            request_logger.error(
                "Edit request failed",
                s3_folder_path=request.s3_folder_path,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

            yield ai_service_pb2.TaskUpdateResponse(
                task_id=task_id,
                title="Edit content",
                status=ai_service_pb2.TaskStatus.FAILED,
                error_message=str(e),
            )

        finally:
            # Clean up callback
            self.task_callbacks.pop(task_id, None)

    async def GetStatus(
        self,
        request: ai_service_pb2.GetStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> ai_service_pb2.GetStatusResponse:
        """Get status of multiple tasks."""
        logger.info(
            "GetStatus request received",
            task_count=len(request.task_ids),
        )

        tasks = {}

        for task_id in request.task_ids:
            # Get task result from Celery
            result = generate_content.AsyncResult(task_id)

            task = ai_service_pb2.Task()

            if result.ready():
                if result.successful():
                    task_result = result.result
                    task.status = ai_service_pb2.TaskStatus.COMPLETED
                    task.result_uri = (
                        task_result.get("output_s3_paths", [""])[0]
                        if task_result.get("output_s3_paths")
                        else ""
                    )
                else:
                    task.status = ai_service_pb2.TaskStatus.FAILED
                    task.error_message = str(result.info)
            elif result.state == "STARTED":
                task.status = ai_service_pb2.TaskStatus.IN_PROGRESS
            else:
                task.status = ai_service_pb2.TaskStatus.PENDING

            tasks[task_id] = task

        return ai_service_pb2.GetStatusResponse(tasks=tasks)


async def serve():
    """Start the gRPC server."""
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    )

    ai_service_pb2_grpc.add_AiServiceServicer_to_server(AiServicer(), server)

    address = f"{settings.grpc_server_host}:{settings.grpc_server_port}"
    server.add_insecure_port(address)

    logger.info("Starting gRPC server", address=address)
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(0)
