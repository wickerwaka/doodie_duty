from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # Camera settings
    camera_index: int = Field(0, description="Camera device index")
    camera_fps: int = Field(30, description="Camera FPS limit")

    # Detection settings
    model_name: str = Field("yolov8n.pt", description="YOLO model to use")
    confidence_threshold: float = Field(0.5, description="Detection confidence threshold")

    # Supervision settings
    alert_delay_seconds: int = Field(5, description="Seconds before triggering alert")
    check_interval_seconds: float = Field(0.5, description="Detection check interval")

    # Web server settings
    host: str = Field("0.0.0.0", description="Web server host")
    port: int = Field(8000, description="Web server port")

    # Database settings
    database_url: str = Field("sqlite+aiosqlite:///doodie_duty.db", description="Database URL")
    cleanup_days: int = Field(30, description="Days to keep events in database")

    # Action settings
    enable_sound_alert: bool = Field(True, description="Enable sound alerts")
    sound_file: Optional[str] = Field(None, description="Custom sound file path")
    enable_file_logging: bool = Field(True, description="Enable file logging")
    log_directory: str = Field("logs", description="Log directory path")
    enable_video_recording: bool = Field(True, description="Enable video recording on alert")
    recording_directory: str = Field("recordings", description="Recording directory path")
    recording_duration: int = Field(30, description="Recording duration in seconds")
    notification_webhook: Optional[str] = Field(None, description="Webhook URL for notifications")
    action_cooldown_seconds: int = Field(60, description="Cooldown between action triggers")

    # Raspberry Pi optimization
    use_lightweight_model: bool = Field(False, description="Use lightweight model for Pi")
    reduce_resolution: bool = Field(False, description="Reduce camera resolution for Pi")
    target_width: int = Field(640, description="Target width when reducing resolution")
    target_height: int = Field(480, description="Target height when reducing resolution")

    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'env_file_encoding': 'utf-8'
    }


def load_config(config_file: Optional[str] = None) -> Settings:
    if config_file and Path(config_file).exists():
        return Settings(_env_file=config_file)
    return Settings()


def save_config(settings: Settings, config_file: str = "config.json"):
    import json
    with open(config_file, 'w') as f:
        json.dump(settings.dict(), f, indent=2)
    print(f"Configuration saved to {config_file}")