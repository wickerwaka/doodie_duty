#!/usr/bin/env python3
"""Simple test server to verify web UI changes"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import json

# Import just the WebApp to get the HTML
from src.web_app import WebApp

app = FastAPI()

# Create a mock supervisor
class MockSupervisor:
    def get_current_status(self):
        return {"status": "test"}

    def get_recent_events(self, limit):
        return []

    def add_event_handler(self, handler):
        pass

    class camera:
        @staticmethod
        def get_frame_sync():
            return None

    class detector:
        @staticmethod
        def is_dog_unsupervised(frame):
            return False, [], []

        @staticmethod
        def draw_detections(frame, detections):
            return frame

# Create WebApp instance to get HTML
web_app = WebApp(MockSupervisor())
html_content = web_app.get_index_html()

@app.get("/", response_class=HTMLResponse)
async def index():
    return html_content

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)

            if command.get("type") == "get_frame":
                # Send mock frame data
                await websocket.send_json({
                    "type": "frame",
                    "data": {
                        "image": "",  # Empty image for testing
                        "dogs": 1,
                        "humans": 0,
                        "is_unsupervised": True
                    }
                })
            elif command.get("type") == "get_status":
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "is_running": True,
                        "duration_unsupervised_seconds": 5.2
                    }
                })
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/events")
async def get_events():
    return []

@app.get("/recordings")
async def get_recordings():
    return {"recordings": []}

@app.post("/start")
async def start():
    return {"message": "Started"}

@app.post("/stop")
async def stop():
    return {"message": "Stopped"}

if __name__ == "__main__":
    import uvicorn
    print("Starting test web UI server...")
    print("Open http://localhost:8000 to test the new video controls")
    print("- Check the video enable/disable checkbox")
    print("- Test different frame rates")
    print("")
    uvicorn.run(app, host="127.0.0.1", port=8000)