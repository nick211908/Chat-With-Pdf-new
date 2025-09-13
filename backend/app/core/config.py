import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Pydantic model for application settings, loaded from environment variables.
    """
    # MongoDB Configuration
    MONGO_CONNECTION_STRING: str

    # Google Gemini API Key
    GOOGLE_GEMINI_API_KEY: str

    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str # This is the service_role key
    SUPABASE_JWT_SECRET: str
    SUPABASE_ANON_KEY: str

    HUGGINGFACE_API_TOKEN:str

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    # --- CORRECTED CONFIGURATION ---
    # This single dictionary now handles all model settings, resolving the conflict.
    # It tells Pydantic where to find the .env file and to ignore extra variables.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Create a single, importable instance of the settings
settings = Settings()

