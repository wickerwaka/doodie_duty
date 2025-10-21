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
            print(f"[ACTION] {self.name} is disabled, skipping")
            return False

        print(f"[ACTION] Triggering {self.name} at {datetime.now().strftime('%H:%M:%S')}")
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
        print(f"[SOUND] Executing sound alert on {self.system}")
        print(f"[SOUND] Event data: dogs={event_data.get('dogs_detected')}, humans={event_data.get('humans_detected')}, duration={event_data.get('duration_unsupervised')}s")

        try:
            if self.system == "Darwin":  # macOS
                if self.sound_file and os.path.exists(self.sound_file):
                    print(f"[SOUND] Playing custom sound: {self.sound_file}")
                    subprocess.run(["afplay", self.sound_file], check=False)
                else:
                    print(f"[SOUND] Using system TTS for alert")
                    subprocess.run(["say", "Alert! Dog detected unsupervised"], check=False)
            elif self.system == "Linux":
                if self.sound_file and os.path.exists(self.sound_file):
                    print(f"[SOUND] Playing custom sound: {self.sound_file}")
                    subprocess.run(["aplay", self.sound_file], check=False)
                else:
                    print(f"[SOUND] Using espeak for alert")
                    subprocess.run(["espeak", "Alert! Dog detected unsupervised"], check=False)
            elif self.system == "Windows":
                import winsound
                if self.sound_file and os.path.exists(self.sound_file):
                    print(f"[SOUND] Playing custom sound: {self.sound_file}")
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME)
                else:
                    print(f"[SOUND] Using system beep")
                    winsound.Beep(1000, 1000)

            print(f"[SOUND] ✓ Sound alert completed successfully")
            return True
        except Exception as e:
            print(f"[SOUND] ✗ Sound alert failed: {e}")
            return False


class FileLogger(ActionTrigger):
    def __init__(self, log_dir: str = "logs"):
        super().__init__("file_logger")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        try:
            log_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.log"
            print(f"[LOG] Writing event to {log_file}")

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "state": event_data.get("state"),
                "dogs_detected": event_data.get("dogs_detected"),
                "humans_detected": event_data.get("humans_detected"),
                "duration_unsupervised": event_data.get("duration_unsupervised")
            }

            print(f"[LOG] Entry: {json.dumps(log_entry)}")

            async with aiofiles.open(log_file, mode='a') as f:
                await f.write(json.dumps(log_entry) + "\n")

            print(f"[LOG] ✓ Event logged successfully")
            return True
        except Exception as e:
            print(f"[LOG] ✗ File logging failed: {e}")
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
            print("[VIDEO] ⚠ Already recording, skipping new recording request")
            return False

        print(f"[VIDEO] Starting {self.duration_seconds}s video recording")
        print(f"[VIDEO] Event: dogs={event_data.get('dogs_detected')}, humans={event_data.get('humans_detected')}")

        try:
            self.is_recording = True
            camera = event_data.get("camera")
            if not camera:
                print("[VIDEO] ✗ No camera provided for recording")
                return False

            filename = self.output_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            print(f"[VIDEO] Recording to: {filename}")

            self.recording_task = asyncio.create_task(
                self._record_video(camera, filename, self.duration_seconds)
            )

            print(f"[VIDEO] ✓ Recording task started")
            return True
        except Exception as e:
            print(f"[VIDEO] ✗ Video recording failed: {e}")
            self.is_recording = False
            return False

    async def _record_video(self, camera, filename: Path, duration: int):
        import cv2

        try:
            # Try browser-compatible codecs in order of preference
            fps = 20
            frame_width = 640
            frame_height = 480

            codecs_to_try = [
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264 (best browser support)
                ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264 alternative
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # MPEG-4 Part 2
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Fallback
            ]

            out = None
            used_codec = None

            for codec_name, fourcc in codecs_to_try:
                print(f"[VIDEO] Trying codec: {codec_name}")
                test_out = cv2.VideoWriter(str(filename), fourcc, fps, (frame_width, frame_height))

                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_name
                    print(f"[VIDEO] ✓ Using codec: {codec_name}")
                    break
                else:
                    test_out.release()
                    print(f"[VIDEO] ✗ Codec {codec_name} failed")

            if out is None:
                print(f"[VIDEO] ✗ All codecs failed")
                return False

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
            print(f"[VIDEO] ✓ Recording completed: {filename}")
            print(f"[VIDEO] Saved {frames_written} frames ({frames_written/fps:.1f}s of video)")
            print(f"[VIDEO] Codec used: {used_codec}")

        except Exception as e:
            print(f"[VIDEO] ✗ Recording error: {e}")
        finally:
            self.is_recording = False
            print(f"[VIDEO] Recording state reset")


class NotificationSender(ActionTrigger):
    def __init__(self, webhook_url: Optional[str] = None):
        super().__init__("notification")
        self.webhook_url = webhook_url

    async def _execute(self, event_data: Dict[str, Any]) -> bool:
        try:
            if self.webhook_url:
                print(f"[WEBHOOK] Sending notification to webhook")
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "text": f"🚨 Doodie Duty Alert! Dog detected unsupervised for {event_data.get('duration_unsupervised', 0):.1f} seconds",
                        "timestamp": datetime.now().isoformat(),
                        "dogs": event_data.get("dogs_detected", 0),
                        "humans": event_data.get("humans_detected", 0)
                    }
                    print(f"[WEBHOOK] Payload: {json.dumps(payload)}")
                    async with session.post(self.webhook_url, json=payload) as resp:
                        if resp.status == 200:
                            print(f"[WEBHOOK] ✓ Notification sent successfully (status={resp.status})")
                            return True
                        else:
                            print(f"[WEBHOOK] ✗ Failed with status {resp.status}")
                            return False
            else:
                print("[WEBHOOK] No webhook URL configured, skipping")
                return False

        except Exception as e:
            print(f"[WEBHOOK] ✗ Notification failed: {e}")
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
        print(f"\n[ACTIONS] ========== TRIGGERING ACTIONS ==========")
        print(f"[ACTIONS] Time: {current_time.strftime('%H:%M:%S')}")
        print(f"[ACTIONS] State: {event_data.get('state')}")
        print(f"[ACTIONS] Dogs: {event_data.get('dogs_detected')}, Humans: {event_data.get('humans_detected')}")
        print(f"[ACTIONS] Duration unsupervised: {event_data.get('duration_unsupervised')}s")
        print(f"[ACTIONS] Active actions: {', '.join(self.actions.keys())}")

        triggered_count = 0
        for name, action in self.actions.items():
            if not action.enabled:
                print(f"[ACTIONS] {name}: disabled")
                continue

            if name in self.last_trigger_time:
                time_since_last = (current_time - self.last_trigger_time[name]).total_seconds()
                if time_since_last < self.cooldown_seconds:
                    print(f"[ACTIONS] {name}: on cooldown ({self.cooldown_seconds - time_since_last:.0f}s remaining)")
                    continue

            try:
                print(f"[ACTIONS] Executing {name}...")
                success = await action.trigger(event_data)
                if success:
                    self.last_trigger_time[name] = current_time
                    print(f"[ACTIONS] {name}: ✓ SUCCESS")
                    triggered_count += 1
                else:
                    print(f"[ACTIONS] {name}: ✗ FAILED")
            except Exception as e:
                print(f"[ACTIONS] {name}: ✗ EXCEPTION - {e}")

        print(f"[ACTIONS] Triggered {triggered_count}/{len(self.actions)} actions")
        print(f"[ACTIONS] ========================================\n")

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