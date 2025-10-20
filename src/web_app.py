from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import asyncio
from datetime import datetime
from typing import List
import io
import base64

from .camera import AsyncCameraCapture
from .detector import DogHumanDetector
from .supervisor import DogSupervisor, SupervisionEvent, SupervisionState


class WebApp:
    def __init__(self, supervisor: DogSupervisor):
        self.app = FastAPI(title="Doodie Duty")
        self.supervisor = supervisor
        self.active_connections: List[WebSocket] = []

        self.setup_routes()
        self.setup_event_handlers()

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return self.get_index_html()

        @self.app.get("/status")
        async def get_status():
            return self.supervisor.get_current_status()

        @self.app.get("/events")
        async def get_events(limit: int = 10):
            events = self.supervisor.get_recent_events(limit)
            return [
                {
                    "state": event.state.value,
                    "timestamp": event.timestamp.isoformat(),
                    "dogs_detected": event.dogs_detected,
                    "humans_detected": event.humans_detected,
                    "duration_unsupervised": event.duration_unsupervised.total_seconds()
                    if event.duration_unsupervised else None
                }
                for event in events
            ]

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    command = json.loads(data)

                    if command.get("type") == "get_frame":
                        await self.send_frame(websocket)
                    elif command.get("type") == "get_status":
                        status = self.supervisor.get_current_status()
                        await websocket.send_json({"type": "status", "data": status})

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                print(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

        @self.app.post("/start")
        async def start_monitoring():
            await self.supervisor.start()
            return {"message": "Monitoring started"}

        @self.app.post("/stop")
        async def stop_monitoring():
            await self.supervisor.stop()
            return {"message": "Monitoring stopped"}

    def setup_event_handlers(self):
        def on_event(event: SupervisionEvent):
            asyncio.create_task(self.broadcast_event(event))

        self.supervisor.add_event_handler(on_event)

    async def send_frame(self, websocket: WebSocket):
        frame = self.supervisor.camera.get_frame_sync()
        if frame is None:
            return

        is_unsupervised, dogs, humans = self.supervisor.detector.is_dog_unsupervised(frame)

        all_detections = dogs + humans
        annotated_frame = self.supervisor.detector.draw_detections(frame, all_detections)

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        await websocket.send_json({
            "type": "frame",
            "data": {
                "image": frame_base64,
                "dogs": len(dogs),
                "humans": len(humans),
                "is_unsupervised": is_unsupervised
            }
        })

    async def broadcast_event(self, event: SupervisionEvent):
        message = {
            "type": "event",
            "data": {
                "state": event.state.value,
                "timestamp": event.timestamp.isoformat(),
                "dogs_detected": event.dogs_detected,
                "humans_detected": event.humans_detected,
                "duration_unsupervised": event.duration_unsupervised.total_seconds()
                if event.duration_unsupervised else None
            }
        }

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    def get_index_html(self) -> str:
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Doodie Duty Monitor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status-bar {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-item {
            display: inline-block;
            margin-right: 30px;
        }
        .status-label {
            font-weight: bold;
            color: #666;
        }
        .video-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        #videoFeed {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .controls {
            margin-top: 20px;
            text-align: center;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
            font-size: 16px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .events {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .event-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .state-idle { color: #999; }
        .state-supervised { color: #4CAF50; }
        .state-unsupervised { color: #ff9800; }
        .state-alert { color: #f44336; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐕 Doodie Duty Monitor</h1>

        <div class="status-bar">
            <div class="status-item">
                <span class="status-label">State:</span>
                <span id="currentState" class="state-idle">Idle</span>
            </div>
            <div class="status-item">
                <span class="status-label">Dogs:</span>
                <span id="dogCount">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Humans:</span>
                <span id="humanCount">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Unsupervised Time:</span>
                <span id="unsupervisedTime">--</span>
            </div>
        </div>

        <div class="video-container">
            <img id="videoFeed" src="" alt="Camera Feed">
            <div class="controls">
                <button id="startBtn" onclick="startMonitoring()">Start Monitoring</button>
                <button id="stopBtn" onclick="stopMonitoring()" disabled>Stop Monitoring</button>
            </div>
        </div>

        <div class="events">
            <h2>Recent Events</h2>
            <div id="eventsList"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let frameInterval = null;
        let isMonitoring = false;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = function() {
                console.log("Connected to server");
                startFrameUpdates();
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === "frame") {
                    updateFrame(data.data);
                } else if (data.type === "event") {
                    addEvent(data.data);
                } else if (data.type === "status") {
                    updateStatus(data.data);
                }
            };

            ws.onclose = function() {
                console.log("Disconnected from server");
                stopFrameUpdates();
                setTimeout(connectWebSocket, 3000);
            };
        }

        function startFrameUpdates() {
            frameInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: "get_frame" }));
                }
            }, 100);
        }

        function stopFrameUpdates() {
            if (frameInterval) {
                clearInterval(frameInterval);
                frameInterval = null;
            }
        }

        function updateFrame(data) {
            const img = document.getElementById("videoFeed");
            img.src = `data:image/jpeg;base64,${data.image}`;

            document.getElementById("dogCount").textContent = data.dogs;
            document.getElementById("humanCount").textContent = data.humans;

            const stateElement = document.getElementById("currentState");
            if (data.dogs === 0) {
                stateElement.textContent = "Idle";
                stateElement.className = "state-idle";
            } else if (data.is_unsupervised) {
                stateElement.textContent = "Unsupervised";
                stateElement.className = "state-unsupervised";
            } else {
                stateElement.textContent = "Supervised";
                stateElement.className = "state-supervised";
            }
        }

        function updateStatus(status) {
            if (status.duration_unsupervised_seconds) {
                const seconds = Math.floor(status.duration_unsupervised_seconds);
                document.getElementById("unsupervisedTime").textContent = `${seconds}s`;
            } else {
                document.getElementById("unsupervisedTime").textContent = "--";
            }

            isMonitoring = status.is_running;
            document.getElementById("startBtn").disabled = isMonitoring;
            document.getElementById("stopBtn").disabled = !isMonitoring;
        }

        function addEvent(event) {
            const eventsList = document.getElementById("eventsList");
            const eventDiv = document.createElement("div");
            eventDiv.className = "event-item";

            const time = new Date(event.timestamp).toLocaleTimeString();
            const stateClass = `state-${event.state}`;

            eventDiv.innerHTML = `
                <strong>${time}</strong> -
                <span class="${stateClass}">${event.state.toUpperCase()}</span> -
                Dogs: ${event.dogs_detected}, Humans: ${event.humans_detected}
                ${event.duration_unsupervised ? ` (${event.duration_unsupervised.toFixed(1)}s)` : ''}
            `;

            eventsList.insertBefore(eventDiv, eventsList.firstChild);

            if (eventsList.children.length > 10) {
                eventsList.removeChild(eventsList.lastChild);
            }
        }

        async function startMonitoring() {
            const response = await fetch("/start", { method: "POST" });
            if (response.ok) {
                console.log("Monitoring started");
                isMonitoring = true;
                document.getElementById("startBtn").disabled = true;
                document.getElementById("stopBtn").disabled = false;
            }
        }

        async function stopMonitoring() {
            const response = await fetch("/stop", { method: "POST" });
            if (response.ok) {
                console.log("Monitoring stopped");
                isMonitoring = false;
                document.getElementById("startBtn").disabled = false;
                document.getElementById("stopBtn").disabled = true;
            }
        }

        async function loadRecentEvents() {
            const response = await fetch("/events");
            if (response.ok) {
                const events = await response.json();
                events.forEach(addEvent);
            }
        }

        connectWebSocket();
        loadRecentEvents();
    </script>
</body>
</html>
        '''