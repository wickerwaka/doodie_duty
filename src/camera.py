import cv2
import asyncio
import threading
import time
from typing import Optional, Callable
import numpy as np
from queue import Queue, Empty


class CameraCapture:
    def __init__(self, camera_index: int = 0, fps_limit: int = 30):
        self.camera_index = camera_index
        self.fps_limit = fps_limit
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.frame_callbacks: list[Callable] = []

    def start(self) -> bool:
        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        print(f"Camera {self.camera_index} started successfully")
        return True

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"Camera {self.camera_index} stopped")

    def _capture_loop(self):
        frame_interval = 1.0 / self.fps_limit
        last_frame_time = 0

        while self.is_running:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)
                continue

            with self.frame_lock:
                self.current_frame = frame

            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass

            for callback in self.frame_callbacks:
                try:
                    callback(frame.copy())
                except Exception as e:
                    print(f"Frame callback error: {e}")

            last_frame_time = current_time

    def get_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def get_frame_nowait(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def add_frame_callback(self, callback: Callable):
        self.frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable):
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)

    def get_camera_info(self) -> dict:
        if not self.cap:
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "backend": self.cap.getBackendName()
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AsyncCameraCapture:
    def __init__(self, camera_index: int = 0, fps_limit: int = 30):
        self.sync_capture = CameraCapture(camera_index, fps_limit)
        self.frame_available = asyncio.Event()

    async def start(self) -> bool:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.sync_capture.start
        )

    async def stop(self):
        await asyncio.get_event_loop().run_in_executor(
            None, self.sync_capture.stop
        )

    async def get_frame(self) -> Optional[np.ndarray]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self.sync_capture.get_frame
        )

    def get_frame_sync(self) -> Optional[np.ndarray]:
        return self.sync_capture.get_frame()

    def get_camera_info(self) -> dict:
        return self.sync_capture.get_camera_info()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()