import asyncio
import uvicorn
import signal
import sys
from pathlib import Path

from src.config import load_config
from src.camera import AsyncCameraCapture
from src.detector import DogHumanDetector
from src.supervisor import DogSupervisor, SupervisionEvent, SupervisionState
from src.web_app import WebApp
from src.database import Database
from src.actions import (
    ActionManager,
    SoundAlert,
    FileLogger,
    VideoRecorder,
    NotificationSender
)


class DoodieDutyApp:
    def __init__(self, config_file: str = None):
        self.config = load_config(config_file)
        self.database = Database(self.config.database_url)
        self.action_manager = ActionManager()
        self.supervisor = None
        self.web_app = None

    async def initialize(self):
        print("Initializing Doodie Duty...")

        await self.database.init_db()

        model_name = "yolov8n.pt" if not self.config.use_lightweight_model else "yolov8n.pt"
        detector = DogHumanDetector(
            model_name=model_name,
            confidence_threshold=self.config.confidence_threshold
        )

        camera = AsyncCameraCapture(
            camera_index=self.config.camera_index,
            fps_limit=self.config.camera_fps
        )

        self.supervisor = DogSupervisor(
            detector=detector,
            camera=camera,
            alert_delay_seconds=self.config.alert_delay_seconds,
            check_interval_seconds=self.config.check_interval_seconds
        )

        self._setup_actions()
        self._setup_event_handlers()

        self.web_app = WebApp(self.supervisor)

        print("Initialization complete!")

    def _setup_actions(self):
        self.action_manager.cooldown_seconds = self.config.action_cooldown_seconds

        if self.config.enable_sound_alert:
            sound_alert = SoundAlert(self.config.sound_file)
            self.action_manager.add_action(sound_alert)

        if self.config.enable_file_logging:
            file_logger = FileLogger(self.config.log_directory)
            self.action_manager.add_action(file_logger)

        if self.config.enable_video_recording:
            video_recorder = VideoRecorder(
                self.config.recording_directory,
                self.config.recording_duration
            )
            self.action_manager.add_action(video_recorder)

        if self.config.notification_webhook:
            notification = NotificationSender(self.config.notification_webhook)
            self.action_manager.add_action(notification)

    def _setup_event_handlers(self):
        async def on_event(event: SupervisionEvent):
            await self.database.log_event(
                state=event.state.value,
                dogs_detected=event.dogs_detected,
                humans_detected=event.humans_detected,
                duration_unsupervised=event.duration_unsupervised.total_seconds()
                if event.duration_unsupervised else None,
                frame_snapshot=event.frame_snapshot,
                detections=event.detections,
                alert_triggered=(event.state == SupervisionState.ALERT)
            )

            if event.state == SupervisionState.ALERT:
                event_data = {
                    "state": event.state.value,
                    "dogs_detected": event.dogs_detected,
                    "humans_detected": event.humans_detected,
                    "duration_unsupervised": event.duration_unsupervised.total_seconds()
                    if event.duration_unsupervised else None,
                    "camera": self.supervisor.camera
                }
                await self.action_manager.trigger_actions(event_data)

        self.supervisor.add_event_handler(on_event)

    async def run(self):
        await self.initialize()

        config = uvicorn.Config(
            app=self.web_app.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        print(f"\n🐕 Doodie Duty is running!")
        print(f"📡 Web interface: http://{self.config.host}:{self.config.port}")
        print(f"📷 Camera: Device {self.config.camera_index}")
        print(f"⏰ Alert delay: {self.config.alert_delay_seconds} seconds")
        print("\nPress Ctrl+C to stop...\n")

        await server.serve()

    async def cleanup(self):
        print("\nShutting down...")
        if self.supervisor:
            await self.supervisor.stop()

        deleted = await self.database.cleanup_old_events(self.config.cleanup_days)
        if deleted > 0:
            print(f"Cleaned up {deleted} old events")

        print("Goodbye! 🐕")


async def main():
    app = DoodieDutyApp()

    def signal_handler(sig, frame):
        print("\nReceived interrupt signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        await app.run()
    except KeyboardInterrupt:
        pass
    finally:
        await app.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Doodie Duty - Dog Supervision Monitor")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera device index",
        default=None
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Web server port",
        default=None
    )

    args = parser.parse_args()

    if args.camera is not None:
        import os
        os.environ["CAMERA_INDEX"] = str(args.camera)
    if args.port is not None:
        import os
        os.environ["PORT"] = str(args.port)

    asyncio.run(main())