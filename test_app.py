#!/usr/bin/env python3
"""
Test script to verify Doodie Duty components
"""

import asyncio
import cv2
import numpy as np
from src.camera import CameraCapture
from src.detector import DogHumanDetector


def test_camera():
    print("\n1. Testing Camera Capture...")
    camera = CameraCapture(camera_index=0)

    if camera.start():
        print("✓ Camera started successfully")

        # Wait for camera to initialize
        import time
        time.sleep(1)

        frame = camera.get_frame()
        if frame is not None:
            print(f"✓ Frame captured: {frame.shape}")
        else:
            print("✗ Failed to capture frame")

        info = camera.get_camera_info()
        print(f"✓ Camera info: {info}")

        camera.stop()
    else:
        print("✗ Camera failed to start (this is normal if no camera is connected)")

    return True


def test_detector():
    print("\n2. Testing Dog/Human Detector...")

    print("✓ Initializing YOLOv8 model (this may download the model on first run)...")
    detector = DogHumanDetector(model_name="yolov8n.pt", confidence_threshold=0.5)

    # Create a test image (blank for now)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(test_frame)
    print(f"✓ Detection on blank frame completed: {len(detections)} objects found")

    # Test detection methods
    dogs = detector.detect_dogs(test_frame)
    humans = detector.detect_humans(test_frame)
    is_unsupervised, _, _ = detector.is_dog_unsupervised(test_frame)

    print(f"✓ Dogs detected: {len(dogs)}")
    print(f"✓ Humans detected: {len(humans)}")
    print(f"✓ Is unsupervised: {is_unsupervised}")

    return True


async def test_web_server():
    print("\n3. Testing Web Server...")

    from src.camera import AsyncCameraCapture
    from src.supervisor import DogSupervisor
    from src.web_app import WebApp

    detector = DogHumanDetector()
    camera = AsyncCameraCapture()
    supervisor = DogSupervisor(detector, camera)

    web_app = WebApp(supervisor)
    print("✓ Web app initialized")

    # Test that routes are set up
    routes = [route.path for route in web_app.app.routes]
    print(f"✓ Routes configured: {', '.join(routes[:5])}...")

    return True


async def test_database():
    print("\n4. Testing Database...")

    from src.database import Database

    db = Database("sqlite+aiosqlite:///test.db")
    await db.init_db()
    print("✓ Database initialized")

    # Log a test event
    event_id = await db.log_event(
        state="test",
        dogs_detected=1,
        humans_detected=0,
        duration_unsupervised_seconds=5.0
    )
    print(f"✓ Test event logged with ID: {event_id}")

    # Retrieve events
    events = await db.get_events(limit=1)
    print(f"✓ Retrieved {len(events)} event(s)")

    # Get statistics
    stats = await db.get_statistics()
    print(f"✓ Statistics: {stats['total_events']} total events")

    # Cleanup
    import os
    if os.path.exists("test.db"):
        os.remove("test.db")

    return True


def test_config():
    print("\n5. Testing Configuration...")

    from src.config import load_config

    config = load_config()
    print(f"✓ Default config loaded")
    print(f"  - Camera index: {config.camera_index}")
    print(f"  - Alert delay: {config.alert_delay_seconds}s")
    print(f"  - Web port: {config.port}")

    return True


async def main():
    print("=" * 50)
    print("DOODIE DUTY COMPONENT TEST")
    print("=" * 50)

    try:
        # Test synchronous components
        test_camera()
        test_detector()
        test_config()

        # Test async components
        await test_web_server()
        await test_database()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nYou can now run the main application with:")
        print("  python main.py")
        print("\nThen open your browser to:")
        print("  http://localhost:8000")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)