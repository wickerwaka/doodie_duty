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
        print(f"\n[MAIN] 🚀 Initializing Doodie Duty...")
        print(f"[MAIN] Config: camera={self.config.camera_index}, alert_delay={self.config.alert_delay_seconds}s")

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

        print(f"[MAIN] ✓ Initialization complete!\n")

    def _setup_actions(self):
        print(f"[MAIN] 🎛️ Setting up action triggers...")
        self.action_manager.cooldown_seconds = self.config.action_cooldown_seconds
        print(f"[MAIN] Action cooldown: {self.config.action_cooldown_seconds}s")

        actions_enabled = []

        if self.config.enable_sound_alert:
            sound_alert = SoundAlert(self.config.sound_file)
            self.action_manager.add_action(sound_alert)
            actions_enabled.append(f"sound_alert ({sound_alert.system})")

        if self.config.enable_file_logging:
            file_logger = FileLogger(self.config.log_directory)
            self.action_manager.add_action(file_logger)
            actions_enabled.append(f"file_logger ({self.config.log_directory})")

        if self.config.enable_video_recording:
            video_recorder = VideoRecorder(
                self.config.recording_directory,
                self.config.recording_duration
            )
            self.action_manager.add_action(video_recorder)
            actions_enabled.append(f"video_recorder ({self.config.recording_duration}s)")

        if self.config.notification_webhook:
            notification = NotificationSender(self.config.notification_webhook)
            self.action_manager.add_action(notification)
            actions_enabled.append("webhook_notification")

        if actions_enabled:
            print(f"[MAIN] Enabled actions: {', '.join(actions_enabled)}")
        else:
            print(f"[MAIN] ⚠️ No actions enabled!")

    def _setup_event_handlers(self):
        async def on_event(event: SupervisionEvent):
            print(f"[MAIN] ⚙️ Processing supervision event: {event.state.value}")
            print(f"[MAIN] Event timestamp: {event.timestamp.strftime('%H:%M:%S')}")

            # Log to database
            try:
                event_id = await self.database.log_event(
                    state=event.state.value,
                    dogs_detected=event.dogs_detected,
                    humans_detected=event.humans_detected,
                    duration_unsupervised_seconds=event.duration_unsupervised.total_seconds()
                    if event.duration_unsupervised else None,
                    frame_snapshot=event.frame_snapshot,
                    detections=event.detections,
                    alert_triggered=(event.state == SupervisionState.ALERT)
                )
                print(f"[MAIN] ✓ Event logged to database (ID: {event_id})")
            except Exception as e:
                print(f"[MAIN] ✗ Database logging failed: {e}")

            # Trigger actions if it's an alert
            if event.state == SupervisionState.ALERT:
                print(f"[MAIN] 🚨 ALERT event - preparing to trigger actions")
                event_data = {
                    "state": event.state.value,
                    "dogs_detected": event.dogs_detected,
                    "humans_detected": event.humans_detected,
                    "duration_unsupervised": event.duration_unsupervised.total_seconds()
                    if event.duration_unsupervised else None,
                    "camera": self.supervisor.camera
                }
                try:
                    await self.action_manager.trigger_actions(event_data)
                except Exception as e:
                    print(f"[MAIN] ✗ Action triggering failed: {e}")
            else:
                print(f"[MAIN] Non-alert event, no actions triggered")

        self.supervisor.add_event_handler(on_event)
        print(f"[MAIN] Event handler registered")

    async def run(self):
        await self.initialize()

        config = uvicorn.Config(
            app=self.web_app.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        print(f"\n[MAIN] ========================================")
        print(f"[MAIN] 🐕 DOODIE DUTY IS RUNNING!")
        print(f"[MAIN] ========================================")
        print(f"[MAIN] 📡 Web interface: http://{self.config.host}:{self.config.port}")
        print(f"[MAIN] 📷 Camera: Device {self.config.camera_index}")
        print(f"[MAIN] ⏰ Alert delay: {self.config.alert_delay_seconds} seconds")
        print(f"[MAIN] 🎛️ Actions: {len(self.action_manager.actions)} configured")
        print(f"[MAIN] ========================================")
        print(f"[MAIN] Press Ctrl+C to stop...\n")

        await server.serve()

    async def cleanup(self):
        print(f"\n[MAIN] 🛑 Shutting down...")
        if self.supervisor:
            await self.supervisor.stop()

        print(f"[MAIN] 🧪 Cleaning up old database events...")
        try:
            deleted = await self.database.cleanup_old_events(self.config.cleanup_days)
            if deleted > 0:
                print(f"[MAIN] ✓ Cleaned up {deleted} old events")
            else:
                print(f"[MAIN] No old events to clean up")
        except Exception as e:
            print(f"[MAIN] ✗ Cleanup failed: {e}")

        print(f"[MAIN] 😭 Goodbye! 🐕")


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