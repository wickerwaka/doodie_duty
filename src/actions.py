import os
import subprocess
import platform
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import asyncio
import json
import aiofiles
from pathlib import Path


class ActionTrigger:
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.last_triggered = None

    async def trigger(self, event_data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        self.last_triggered = datetime.now()
        return await self._execute(event_data)

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        raise NotImplementedError


class SoundAlert(ActionTrigger):
    def __init__(self, sound_file: Optional[str] = None):
        super().__init__("sound_alert")
        self.sound_file = sound_file
        self.system = platform.system()

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        try:
            if self.system == "Darwin":  # macOS
                if self.sound_file and os.path.exists(self.sound_file):
                    subprocess.run(["afplay", self.sound_file], check=False)
                else:
                    subprocess.run(["say", "Alert! Dog detected unsupervised"], check=False)
            elif self.system == "Linux":
                if self.sound_file and os.path.exists(self.sound_file):
                    subprocess.run(["aplay", self.sound_file], check=False)
                else:
                    subprocess.run(["espeak", "Alert! Dog detected unsupervised"], check=False)
            elif self.system == "Windows":
                import winsound
                if self.sound_file and os.path.exists(self.sound_file):
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME)
                else:
                    winsound.Beep(1000, 1000)

            print(f"Sound alert triggered at {datetime.now()}")
            return True
        except Exception as e:
            print(f"Sound alert failed: {e}")
            return False


class FileLogger(ActionTrigger):
    def __init__(self, log_dir: str = "logs"):
        super().__init__("file_logger")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        try:
            log_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.log"

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "state": event_data.get("state"),
                "dogs_detected": event_data.get("dogs_detected"),
                "humans_detected": event_data.get("humans_detected"),
                "duration_unsupervised": event_data.get("duration_unsupervised")
            }

            async with aiofiles.open(log_file, mode='a') as f:
                await f.write(json.dumps(log_entry) + "\n")

            return True
        except Exception as e:
            print(f"File logging failed: {e}")
            return False


class VideoRecorder(ActionTrigger):
    def __init__(self, output_dir: str = "recordings", duration_seconds: int = 30):
        super().__init__("video_recorder")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.duration_seconds = duration_seconds
        self.is_recording = False
        self.recording_task = None

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        if self.is_recording:
            print("Already recording, skipping new recording request")
            return False

        try:
            self.is_recording = True
            camera = event_data.get("camera")
            if not camera:
                print("No camera provided for recording")
                return False

            filename = self.output_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.recording_task = asyncio.create_task(
                self._record_video(camera, filename, self.duration_seconds)
            )

            return True
        except Exception as e:
            print(f"Video recording failed: {e}")
            self.is_recording = False
            return False

    async def _record_video(self, camera, filename: Path, duration: int):
        import cv2

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20
            frame_width = 640
            frame_height = 480

            out = cv2.VideoWriter(str(filename), fourcc, fps, (frame_width, frame_height))

            start_time = datetime.now()
            frames_written = 0

            while (datetime.now() - start_time).total_seconds() < duration:
                frame = camera.get_frame_sync()
                if frame is not None:
                    resized = cv2.resize(frame, (frame_width, frame_height))
                    out.write(resized)
                    frames_written += 1

                await asyncio.sleep(1.0 / fps)

            out.release()
            print(f"Recording saved: {filename} ({frames_written} frames)")

        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            self.is_recording = False


class NotificationSender(ActionTrigger):
    def __init__(self, webhook_url: Optional[str] = None):
        super().__init__("notification")
        self.webhook_url = webhook_url

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        try:
            if self.webhook_url:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": f"🚨 Doodie Duty Alert! Dog detected unsupervised for {event_data.get('duration_unsupervised', 0):.1f} seconds",
                        "timestamp": datetime.now().isoformat(),
                        "dogs": event_data.get("dogs_detected", 0),
                        "humans": event_data.get("humans_detected", 0)
                    }
                    async with session.post(self.webhook_url, json=payload) as resp:
                        if resp.status == 200:
                            print("Notification sent successfully")
                            return True
            else:
                print("No webhook URL configured")
                return False

        except Exception as e:
            print(f"Notification failed: {e}")
            return False


class ActionManager:
    def __init__(self):
        self.actions: Dict[str, ActionTrigger] = {}
        self.cooldown_seconds = 60
        self.last_trigger_time = {}

    def add_action(self, action: ActionTrigger):
        self.actions[action.name] = action

    def remove_action(self, name: str):
        if name in self.actions:
            del self.actions[name]

    def enable_action(self, name: str):
        if name in self.actions:
            self.actions[name].enabled = True

    def disable_action(self, name: str):
        if name in self.actions:
            self.actions[name].enabled = False

    async def trigger_actions(self, event_data: Dict[str, Any]):
        current_time = datetime.now()

        for name, action in self.actions.items():
            if not action.enabled:
                continue

            if name in self.last_trigger_time:
                time_since_last = (current_time - self.last_trigger_time[name]).total_seconds()
                if time_since_last < self.cooldown_seconds:
                    print(f"Action {name} on cooldown ({self.cooldown_seconds - time_since_last:.0f}s remaining)")
                    continue

            try:
                success = await action.trigger(event_data)
                if success:
                    self.last_trigger_time[name] = current_time
                    print(f"Action {name} triggered successfully")
            except Exception as e:
                print(f"Action {name} failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "actions": {
                name: {
                    "enabled": action.enabled,
                    "last_triggered": action.last_triggered.isoformat() if action.last_triggered else None
                }
                for name, action in self.actions.items()
            },
            "cooldown_seconds": self.cooldown_seconds
        }