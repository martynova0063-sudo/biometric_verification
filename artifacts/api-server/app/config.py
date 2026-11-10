from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ИС УКЗ · HEALTH_ID API"
    app_version: str = "1.0.0"
    database_url: str = "sqlite:///./ukz_health_id.db"
    session_secret: str
    model_version: str = "1.0.0-research"
    api_prefix: str = "/api"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def sync_database_url(self) -> str:
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()