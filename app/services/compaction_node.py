"""Compaction node for LangGraph router — compacts old messages into a rolling summary."""

import uuid
from typing import Optional, List

from langchain_core.messages import SystemMessage, RemoveMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from app.services.llm_service import LLMService
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Issue M1 fix: cache tiktoken encoder at module level ---
_tiktoken_encoder = None


def _get_tiktoken_encoder():
    """Get cached tiktoken encoder, or None if tiktoken is not installed."""
    global _tiktoken_encoder
    if _tiktoken_encoder is not None:
        return _tiktoken_encoder
    try:
        import tiktoken

        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        return _tiktoken_encoder
    except ImportError:
        return None


# Prefix used to identify compaction summary messages
COMPACTION_SUMMARY_PREFIX = (
    "[Conversation Summary — Earlier messages have been summarized]\n\n"
)

COMPACTION_PROMPT = """Summarize this conversation history for an AI assistant's context.

Preserve with exact values (do not paraphrase):
- File paths, S3 paths, URLs mentioned
- Code structure decisions (frameworks, patterns, file organization)
- User's stated preferences and requirements
- Error messages or issues discussed
- Any numbers, dates, or configuration values
- Key decisions and conclusions reached
- Task results: what was generated and where it is stored

Summarize in the same language as the conversation.
Keep chronological order. Be concise but preserve all important details.
Target: 800-1200 tokens."""


def estimate_token_count(messages: list[BaseMessage]) -> int:
    """Estimate total tokens in messages.

    Uses tiktoken if available, falls back to heuristic.
    The same function is used for BOTH threshold checking and split-point
    calculation (Issue 5 fix).
    """
    encoder = _get_tiktoken_encoder()
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if encoder is not None:
            total += len(encoder.encode(content))
        else:
            # Fallback heuristic: ~3 chars per token for mixed EN/CJK
            total += len(content) // 3
    return total


def _is_compaction_summary(msg: BaseMessage) -> bool:
    """Check if a message is a compaction summary."""
    if not isinstance(msg, SystemMessage):
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return content.startswith(COMPACTION_SUMMARY_PREFIX)


def find_compaction_split(messages: list[BaseMessage], threshold: int) -> Optional[int]:
    """Find the split point: keep recent messages whose tokens < 40% of threshold.

    Returns index where old messages end (exclusive), or None if no compaction needed.

    Issue 5 fix: uses estimate_token_count() consistently for both threshold
    check and split-point budget calculation.
    """
    total_tokens = estimate_token_count(messages)

    if total_tokens <= threshold:
        return None  # No compaction needed

    # Keep recent messages that fit in 40% of threshold
    keep_budget = int(threshold * 0.4)

    # Walk backwards from end, accumulating tokens using the SAME estimator
    keep_start = len(messages)
    running_tokens = 0
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = estimate_token_count([messages[i]])  # Issue 5: same estimator
        if running_tokens + msg_tokens > keep_budget:
            break
        running_tokens += msg_tokens
        keep_start = i

    # Must keep at least the last message (current user input)
    if keep_start >= len(messages):
        keep_start = len(messages) - 1

    # Must have something to summarize (at least 2 messages)
    if keep_start < 2:
        return None

    return keep_start


async def compaction_node(state: "RouterState", config: RunnableConfig) -> dict:
    """LangGraph node: compact old messages into a summary if over token threshold.

    Issue M3 fix: type hint uses RouterState (via string annotation to avoid
    circular import — RouterState is defined in router_graph.py).

    Returns:
        Dict with messages to add/remove, or empty dict if no compaction needed.
    """
    messages = state.get("messages", [])

    # --- Issue 6 fix: raise minimum to 30 to skip token estimation on short convos ---
    if len(messages) < 30:
        return {}

    threshold = settings.chat_compaction_token_threshold
    split_idx = find_compaction_split(messages, threshold)

    if split_idx is None:
        return {}

    # Messages to summarize
    old_messages = messages[:split_idx]

    request_id = config.get("configurable", {}).get("request_id", "unknown")
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")

    logger.info(
        "Compaction triggered",
        request_id=request_id,
        thread_id=thread_id,
        total_messages=len(messages),
        messages_to_summarize=len(old_messages),
        messages_to_keep=len(messages) - split_idx,
        estimated_tokens=estimate_token_count(messages),
    )

    try:
        # --- Issue 7 fix: archive old messages before deletion ---
        await _archive_messages(thread_id, old_messages, request_id)

        # Build text to summarize
        summary_parts = []
        for msg in old_messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif hasattr(msg, "type") and msg.type == "human":
                role = "user"
            else:
                role = "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            summary_parts.append(f"{role}: {content}")

        text_to_summarize = "\n\n".join(summary_parts)

        # LLM summarize
        llm_service = LLMService.get_instance(component="chat")
        llm_messages = LLMService.create_messages(
            system_prompt=COMPACTION_PROMPT,
            user_prompt=f"Conversation to summarize:\n\n{text_to_summarize}",
        )

        summary_text = await llm_service.generate(messages=llm_messages)

        # --- Issue M2 fix: proportional quality check ---
        # Summary should be at least 5% of input length (by chars) and >= 100 chars
        min_length = max(100, len(text_to_summarize) // 20)
        if not summary_text or len(summary_text.strip()) < min_length:
            logger.warning(
                "Compaction summary too short, skipping",
                request_id=request_id,
                summary_length=len(summary_text) if summary_text else 0,
                min_required=min_length,
            )
            return {}

        # Build state update: remove old messages, add summary
        messages_update = []

        # --- Issue 10 fix: raise on missing ID instead of silent skip ---
        for msg in old_messages:
            if not hasattr(msg, "id") or not msg.id:
                raise ValueError(
                    f"Message missing ID — cannot compact. "
                    f"Type={type(msg).__name__}, content_preview={str(msg.content)[:80]}"
                )
            messages_update.append(RemoveMessage(id=msg.id))

        # --- Issue 4 fix: unique ID per compaction event ---
        summary_id = f"compaction-summary-{uuid.uuid4()}"
        messages_update.append(
            SystemMessage(
                content=COMPACTION_SUMMARY_PREFIX + summary_text.strip(),
                id=summary_id,
            )
        )

        logger.info(
            "Compaction completed",
            request_id=request_id,
            thread_id=thread_id,
            messages_removed=len(old_messages),
            summary_id=summary_id,
            summary_length=len(summary_text),
            estimated_summary_tokens=estimate_token_count(
                [SystemMessage(content=summary_text)]
            ),
        )

        return {"messages": messages_update}

    except Exception as e:
        logger.error(
            "Compaction failed, continuing without compaction",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        # Graceful degradation: if compaction fails, just continue
        return {}


async def _archive_messages(
    thread_id: str,
    messages: List[BaseMessage],
    request_id: str,
) -> None:
    """Archive messages to compaction_archive table before deletion (Issue 7 fix).

    This provides a recovery path if a compaction summary is poor quality.
    Archives are keyed by (thread_id, compaction_timestamp) and stored as JSONB.

    If archiving fails, we log a warning but do NOT abort compaction — the
    archive is a safety net, not a hard dependency.
    """
    try:
        import json

        import psycopg
        from datetime import datetime, timezone

        serialized = []
        for msg in messages:
            serialized.append(
                {
                    "type": type(msg).__name__,
                    "id": msg.id,
                    "content": msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content),
                }
            )

        async with await psycopg.AsyncConnection.connect(
            settings.database_url
        ) as conn:
            await conn.execute(
                """
                INSERT INTO compaction_archive (thread_id, archived_at, messages, request_id)
                VALUES (%s, %s, %s::jsonb, %s)
                """,
                (
                    thread_id,
                    datetime.now(timezone.utc),
                    json.dumps(serialized),
                    request_id,
                ),
            )
            await conn.commit()

        logger.info(
            "Messages archived before compaction",
            thread_id=thread_id,
            request_id=request_id,
            message_count=len(messages),
        )
    except Exception as e:
        logger.warning(
            "Failed to archive messages (compaction will proceed anyway)",
            thread_id=thread_id,
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
