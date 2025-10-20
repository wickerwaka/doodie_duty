#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced logging output
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime
from src.camera import AsyncCameraCapture
from src.detector import DogHumanDetector, Detection
from src.supervisor import DogSupervisor, SupervisionState, SupervisionEvent
from src.actions import ActionManager, SoundAlert, FileLogger, VideoRecorder
from src.config import load_config


async def simulate_detection_scenario():
    print("=" * 60)
    print("TESTING DOODIE DUTY LOGGING")
    print("=" * 60)

    # Load config
    config = load_config()

    # Initialize components
    print("\n[TEST] Setting up components...")
    detector = DogHumanDetector(confidence_threshold=0.3)
    camera = AsyncCameraCapture(camera_index=0)

    # Setup action manager
    action_manager = ActionManager()
    action_manager.cooldown_seconds = 5  # Shorter for testing

    # Add actions
    sound_alert = SoundAlert()
    file_logger = FileLogger("test_logs")
    action_manager.add_action(sound_alert)
    action_manager.add_action(file_logger)

    print(f"[TEST] Actions configured: {', '.join(action_manager.actions.keys())}")

    # Create supervisor
    supervisor = DogSupervisor(
        detector=detector,
        camera=camera,
        alert_delay_seconds=3,  # Short delay for testing
        check_interval_seconds=1.0
    )

    # Add event handler to trigger actions
    async def test_event_handler(event: SupervisionEvent):
        print(f"[TEST] 📧 Event handler received: {event.state.value}")

        if event.state == SupervisionState.ALERT:
            print(f"[TEST] 🚨 Triggering actions for ALERT event...")
            event_data = {
                "state": event.state.value,
                "dogs_detected": event.dogs_detected,
                "humans_detected": event.humans_detected,
                "duration_unsupervised": event.duration_unsupervised.total_seconds()
                if event.duration_unsupervised else None,
                "camera": camera
            }
            await action_manager.trigger_actions(event_data)

    supervisor.add_event_handler(test_event_handler)

    print(f"[TEST] 🎯 Starting monitoring for 15 seconds...")
    print(f"[TEST] 📷 Look at the camera to see human detection")
    print(f"[TEST] 🐕 Any dogs detected will trigger supervision logic")
    print(f"[TEST] ⏰ If unsupervised for 3+ seconds, actions will trigger")

    try:
        # Start supervisor
        await supervisor.start()

        # Monitor for a short time
        await asyncio.sleep(15)

        print(f"\n[TEST] 📊 Final status:")
        status = supervisor.get_current_status()
        print(f"[TEST] Current state: {status['state']}")
        print(f"[TEST] Events recorded: {status['last_event_count']}")

        if status['duration_unsupervised_seconds']:
            print(f"[TEST] Unsupervised time: {status['duration_unsupervised_seconds']:.1f}s")

        # Show recent events
        events = supervisor.get_recent_events(5)
        if events:
            print(f"\n[TEST] Recent events:")
            for event in events[-3:]:
                print(f"[TEST]   {event.timestamp.strftime('%H:%M:%S')} - {event.state.value} (dogs:{event.dogs_detected}, humans:{event.humans_detected})")

    except KeyboardInterrupt:
        print(f"\n[TEST] Test interrupted by user")

    finally:
        await supervisor.stop()
        print(f"\n[TEST] ✅ Test completed")


async def test_action_triggers():
    """Test action triggers directly"""
    print(f"\n[TEST] 🧪 Testing action triggers directly...")

    action_manager = ActionManager()
    action_manager.cooldown_seconds = 1  # Very short for testing

    # Add actions
    sound_alert = SoundAlert()
    file_logger = FileLogger("test_logs")
    action_manager.add_action(sound_alert)
    action_manager.add_action(file_logger)

    # Simulate alert event data
    test_event_data = {
        "state": "alert",
        "dogs_detected": 1,
        "humans_detected": 0,
        "duration_unsupervised": 5.5,
        "camera": None  # No camera for this test
    }

    print(f"[TEST] Triggering actions with test data...")
    await action_manager.trigger_actions(test_event_data)

    print(f"[TEST] Action trigger test completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Doodie Duty logging")
    parser.add_argument("--actions-only", action="store_true", help="Test actions only")
    args = parser.parse_args()

    if args.actions_only:
        asyncio.run(test_action_triggers())
    else:
        asyncio.run(simulate_detection_scenario())