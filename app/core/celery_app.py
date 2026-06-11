"""Celery configuration and app initialization."""

import os

from celery import Celery
from kombu import Exchange, Queue

from app.core.config import settings
from app.core.logging import setup_logging

# Setup structured logging first
setup_logging()

# Configure LangSmith tracing before any LangChain imports
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

# Create Celery app
celery_app = Celery(
    "ai_backend",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=600,  # 10 minutes hard limit
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,  # Each worker takes only 1 task at a time
    worker_max_tasks_per_child=1,  # Restart worker after each task (matches docker-compose.yml)
    task_acks_late=True,  # Don't acknowledge task until it's completed - ensures parallel distribution
    worker_hijack_root_logger=False,  # Don't let Celery hijack logging configuration
    worker_log_format="%(message)s",  # Use simple format since we have structured logging
    worker_task_log_format="%(message)s",
)

# Define queues
default_exchange = Exchange("default", type="direct")
generate_exchange = Exchange("generate", type="direct")
edit_exchange = Exchange("edit", type="direct")

celery_app.conf.task_queues = (
    Queue("default", default_exchange, routing_key="default"),
    Queue("generate", generate_exchange, routing_key="generate"),
    Queue("edit", edit_exchange, routing_key="edit"),
)

# Route tasks to appropriate queues
celery_app.conf.task_routes = {
    "app.services.tasks.generate_content": {"queue": "generate"},
    "app.services.tasks.edit_content": {"queue": "edit"},
}

# Import tasks to register them
celery_app.autodiscover_tasks(["app.services"])
