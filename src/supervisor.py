import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Callable
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from .detector import DogHumanDetector, Detection
from .camera import AsyncCameraCapture


class SupervisionState(Enum):
    IDLE = "idle"  # No dog present
    SUPERVISED = "supervised"  # Dog with human
    UNSUPERVISED = "unsupervised"  # Dog without human
    ALERT = "alert"  # Unsupervised for too long


@dataclass
class SupervisionEvent:
    state: SupervisionState
    timestamp: datetime
    dogs_detected: int
    humans_detected: int
    duration_unsupervised: Optional[timedelta] = None
    frame_snapshot: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)


class DogSupervisor:
    def __init__(
        self,
        detector: DogHumanDetector,
        camera: AsyncCameraCapture,
        alert_delay_seconds: int = 5,
        check_interval_seconds: float = 0.5
    ):
        self.detector = detector
        self.camera = camera
        self.alert_delay_seconds = alert_delay_seconds
        self.check_interval_seconds = check_interval_seconds

        self.current_state = SupervisionState.IDLE
        self.unsupervised_start_time: Optional[datetime] = None
        self.last_event_time = datetime.now()

        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None

        self.event_handlers: List[Callable[[SupervisionEvent], None]] = []
        self.state_change_handlers: List[Callable[[SupervisionState, SupervisionState], None]] = []

        self.event_history: List[SupervisionEvent] = []
        self.max_history_size = 1000

    async def start(self):
        if self.is_running:
            return

        if not await self.camera.start():
            raise RuntimeError("Failed to start camera")

        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        print(f"[SUPERVISOR] 🚀 Dog supervisor started")
        print(f"[SUPERVISOR] Camera: {self.camera.get_camera_info()}")
        print(f"[SUPERVISOR] Alert delay: {self.alert_delay_seconds}s")
        print(f"[SUPERVISOR] Check interval: {self.check_interval_seconds}s")
        print(f"[SUPERVISOR] Event handlers: {len(self.event_handlers)}")
        print(f"[SUPERVISOR] State change handlers: {len(self.state_change_handlers)}")

    async def stop(self):
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        await self.camera.stop()
        print(f"\n[SUPERVISOR] 🛑 Dog supervisor stopped")
        print(f"[SUPERVISOR] Total events recorded: {len(self.event_history)}")
        if self.unsupervised_start_time:
            duration = (datetime.now() - self.unsupervised_start_time).total_seconds()
            print(f"[SUPERVISOR] Final unsupervised duration: {duration:.1f}s")

    async def _monitor_loop(self):
        while self.is_running:
            try:
                await self._check_supervision()
                await asyncio.sleep(self.check_interval_seconds)
            except Exception as e:
                print(f"[SUPERVISOR] ❌ Monitor loop error: {e}")
                await asyncio.sleep(1)

    async def _check_supervision(self):
        frame = await self.camera.get_frame()
        if frame is None:
            return

        is_unsupervised, dogs, humans = await asyncio.get_event_loop().run_in_executor(
            None, self.detector.is_dog_unsupervised, frame
        )

        new_state = self._determine_state(is_unsupervised, len(dogs), len(humans))

        # Only log detection details if there are detections or state changes
        if len(dogs) > 0 or len(humans) > 0 or new_state != self.current_state:
            print(f"[DETECT] Dogs: {len(dogs)}, Humans: {len(humans)}, State: {new_state.value}")

        if new_state != self.current_state:
            self._handle_state_change(self.current_state, new_state, dogs, humans, frame)

        elif new_state == SupervisionState.UNSUPERVISED:
            self._check_alert_condition(dogs, humans, frame)

    def _determine_state(self, is_unsupervised: bool, dog_count: int, human_count: int) -> SupervisionState:
        if dog_count == 0:
            return SupervisionState.IDLE
        elif is_unsupervised:
            return SupervisionState.UNSUPERVISED
        else:
            return SupervisionState.SUPERVISED

    def _handle_state_change(
        self,
        old_state: SupervisionState,
        new_state: SupervisionState,
        dogs: List[Detection],
        humans: List[Detection],
        frame: np.ndarray
    ):
        timestamp = datetime.now()
        print(f"\n[STATE] ========== STATE CHANGE ==========")
        print(f"[STATE] Time: {timestamp.strftime('%H:%M:%S')}")
        print(f"[STATE] Transition: {old_state.value} → {new_state.value}")
        print(f"[STATE] Dogs detected: {len(dogs)}")
        print(f"[STATE] Humans detected: {len(humans)}")

        if dogs:
            for i, dog in enumerate(dogs):
                print(f"[STATE]   Dog {i+1}: confidence={dog.confidence:.2f}, bbox={dog.bbox}")

        if humans:
            for i, human in enumerate(humans):
                print(f"[STATE]   Human {i+1}: confidence={human.confidence:.2f}, bbox={human.bbox}")

        if new_state == SupervisionState.UNSUPERVISED:
            self.unsupervised_start_time = timestamp
            print(f"[STATE] ⚠️  Starting unsupervised timer")
        else:
            if self.unsupervised_start_time:
                duration = (timestamp - self.unsupervised_start_time).total_seconds()
                print(f"[STATE] ✅ Ending unsupervised period (lasted {duration:.1f}s)")
            self.unsupervised_start_time = None

        event = SupervisionEvent(
            state=new_state,
            timestamp=datetime.now(),
            dogs_detected=len(dogs),
            humans_detected=len(humans),
            frame_snapshot=frame.copy() if frame is not None else None,
            detections=dogs + humans
        )

        self._trigger_event(event)

        print(f"[STATE] Notifying {len(self.state_change_handlers)} state change handlers")
        for i, handler in enumerate(self.state_change_handlers):
            try:
                handler(old_state, new_state)
                print(f"[STATE] Handler {i+1}: ✓")
            except Exception as e:
                print(f"[STATE] Handler {i+1}: ✗ Error - {e}")

        self.current_state = new_state
        print(f"[STATE] Current state updated to: {new_state.value}")
        print(f"[STATE] ========================================\n")

    def _check_alert_condition(self, dogs: List[Detection], humans: List[Detection], frame: np.ndarray):
        if self.unsupervised_start_time is None:
            return

        duration_unsupervised = datetime.now() - self.unsupervised_start_time

        if (duration_unsupervised.total_seconds() >= self.alert_delay_seconds
            and self.current_state != SupervisionState.ALERT):

            print(f"\n[ALERT] 🚨 ALERT TRIGGERED! 🚨")
            print(f"[ALERT] Dog has been unsupervised for {duration_unsupervised.total_seconds():.1f} seconds")
            print(f"[ALERT] Threshold: {self.alert_delay_seconds} seconds")
            print(f"[ALERT] Dogs: {len(dogs)}, Humans: {len(humans)}")

            event = SupervisionEvent(
                state=SupervisionState.ALERT,
                timestamp=datetime.now(),
                dogs_detected=len(dogs),
                humans_detected=len(humans),
                duration_unsupervised=duration_unsupervised,
                frame_snapshot=frame.copy() if frame is not None else None,
                detections=dogs + humans
            )

            self._trigger_event(event)
            self.current_state = SupervisionState.ALERT
            print(f"[ALERT] State changed to ALERT")
            print(f"[ALERT] Event triggered for action handlers\n")

    def _trigger_event(self, event: SupervisionEvent):
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

        print(f"[EVENT] Triggering event: {event.state.value}")
        print(f"[EVENT] Notifying {len(self.event_handlers)} event handlers")

        for i, handler in enumerate(self.event_handlers):
            try:
                handler(event)
                print(f"[EVENT] Handler {i+1}: ✓")
            except Exception as e:
                print(f"[EVENT] Handler {i+1}: ✗ Error - {e}")

    def add_event_handler(self, handler: Callable[[SupervisionEvent], None]):
        self.event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[SupervisionEvent], None]):
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)

    def add_state_change_handler(self, handler: Callable[[SupervisionState, SupervisionState], None]):
        self.state_change_handlers.append(handler)

    def get_current_status(self) -> dict:
        duration_unsupervised = None
        if self.unsupervised_start_time:
            duration_unsupervised = (datetime.now() - self.unsupervised_start_time).total_seconds()

        return {
            "state": self.current_state.value,
            "is_running": self.is_running,
            "duration_unsupervised_seconds": duration_unsupervised,
            "camera_info": self.camera.get_camera_info(),
            "alert_delay_seconds": self.alert_delay_seconds,
            "last_event_count": len(self.event_history)
        }

    def get_recent_events(self, limit: int = 10) -> List[SupervisionEvent]:
        return self.event_history[-limit:]

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()