"""Runtime configuration and environment parameter parsing."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Operational Parameters
    max_research_iterations: int = Field(default=2, alias="MAX_RESEARCH_ITERATIONS")
    max_search_results_per_query: int = Field(default=4, alias="MAX_SEARCH_RESULTS_PER_QUERY")
    max_pages_to_fetch: int = Field(default=3, alias="MAX_PAGES_TO_FETCH")
    request_timeout_seconds: float = Field(default=10.0, alias="REQUEST_TIMEOUT_SECONDS")
    max_page_content_bytes: int = Field(default=500_000, alias="MAX_PAGE_CONTENT_BYTES")
    max_extracted_text_chars: int = Field(default=12_000, alias="MAX_EXTRACTED_TEXT_CHARS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
