"""Configuration management for the AI backend service."""

import os
from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_GEMINI = "google_gemini"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(
        "https://indieapp.openai.azure.com/", env="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_api_version: str = Field(
        "2024-12-01-preview", env="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment_name: Optional[str] = Field(
        "o3", env="AZURE_OPENAI_DEPLOYMENT_NAME"
    )

    # Azure OpenAI Image Configuration (for image generation)
    azure_openai_image_api_key: Optional[str] = Field(
        None, env="AZURE_OPENAI_IMAGE_API_KEY"
    )
    azure_openai_image_endpoint: Optional[str] = Field(
        None, env="AZURE_OPENAI_IMAGE_ENDPOINT"
    )
    azure_openai_image_deployment_name: Optional[str] = Field(
        "gpt-image-1", env="AZURE_OPENAI_IMAGE_DEPLOYMENT_NAME"
    )
    azure_openai_image_api_version: str = Field(
        "2025-04-01-preview", env="AZURE_OPENAI_IMAGE_API_VERSION"
    )
    azure_openai_image_edit_timeout: int = Field(
        300, env="AZURE_OPENAI_IMAGE_EDIT_TIMEOUT"
    )

    # OpenAI Image Generation Configuration
    openai_image_api_key: Optional[str] = Field(
        None, env="OPENAI_IMAGE_API_KEY"
    )  # Can be same as OPENAI_API_KEY
    image_generation_provider: str = Field(
        "openai", env="IMAGE_GENERATION_PROVIDER"
    )  # "azure", "openai", or "google"
    image_generation_model: str = Field(
        "gpt-image-1", env="IMAGE_GENERATION_MODEL"
    )  # "gpt-image-1", "dall-e-2", "dall-e-3"

    # PPT Generation Configuration
    ppt_provider: str = Field(
        "nanobanana", env="PPT_PROVIDER"
    )  # "nanobanana" (image-based) or "presenton" (pptx-based)

    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    # Google Cloud Vertex AI Configuration (for Gemini image generation)
    google_cloud_project_id: Optional[str] = Field(
        None, env="GOOGLE_CLOUD_PROJECT_ID"
    )
    google_cloud_location: str = Field("global", env="GOOGLE_CLOUD_LOCATION")
    google_application_credentials: Optional[str] = Field(
        None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # AWS Configuration
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-2", env="AWS_REGION")
    s3_bucket_name: str = Field("indieapp-artifact", env="S3_BUCKET_NAME")

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # PostgreSQL Configuration (for chat memory)
    database_url: str = Field(
        "postgresql://chat_user:dev_password_123@localhost:5432/chat_db",
        env="DATABASE_URL",
    )

    # gRPC Configuration
    grpc_server_host: str = Field("0.0.0.0", env="GRPC_SERVER_HOST")
    grpc_server_port: int = Field(50051, env="GRPC_SERVER_PORT")

    # LLM Configuration
    llm_provider: LLMProvider = Field(LLMProvider.AZURE_OPENAI, env="LLM_PROVIDER")
    llm_model: str = Field("o3", env="LLM_MODEL")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")

    # LangSmith Configuration (for agent observability)
    langchain_tracing_v2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("indieapp-ai", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT"
    )

    # Performance
    max_context_tokens: int = Field(8000, env="MAX_CONTEXT_TOKENS")
    streaming_chunk_size: int = Field(50, env="STREAMING_CHUNK_SIZE")

    # Compaction settings (Issue #70)
    chat_compaction_token_threshold: int = Field(
        50000, env="CHAT_COMPACTION_TOKEN_THRESHOLD"
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file

    def validate_llm_config(self) -> None:
        """Validate that the necessary API keys are present for the selected provider."""
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        elif self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required when using Anthropic provider"
            )
        elif self.llm_provider == LLMProvider.AZURE_OPENAI:
            if not self.azure_openai_api_key:
                raise ValueError(
                    "Azure OpenAI API key is required when using Azure OpenAI provider"
                )
            if not self.azure_openai_endpoint:
                raise ValueError(
                    "Azure OpenAI endpoint is required when using Azure OpenAI provider"
                )
            if not self.azure_openai_deployment_name:
                raise ValueError(
                    "Azure OpenAI deployment name is required when using Azure OpenAI provider"
                )
        elif self.llm_provider == LLMProvider.AWS_BEDROCK:
            # AWS credentials are already validated as required fields
            pass
        elif self.llm_provider == LLMProvider.GOOGLE_GEMINI and not self.google_api_key:
            raise ValueError(
                "Google API key is required when using Google Gemini provider"
            )


# Global settings instance
settings = Settings()
settings.validate_llm_config()
