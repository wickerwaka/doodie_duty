from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import asyncio
from datetime import datetime
from typing import List
import io
import base64
import os
from pathlib import Path

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

        @self.app.get("/recordings")
        async def get_recordings():
            return await self.get_recordings_list()

        @self.app.get("/recordings/{filename}")
        async def get_recording(request: Request, filename: str):
            return await self.serve_recording(request, filename)

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

    async def get_recordings_list(self):
        """Get list of all recording files with metadata"""
        recordings_dir = Path("recordings")
        if not recordings_dir.exists():
            return {"recordings": []}

        recordings = []
        for file_path in recordings_dir.glob("alert_*.mp4"):
            try:
                stat = file_path.stat()

                # Parse timestamp from filename (alert_YYYYMMDD_HHMMSS.mp4)
                name_parts = file_path.stem.split("_")
                if len(name_parts) >= 3:
                    date_str = name_parts[1]
                    time_str = name_parts[2]
                    timestamp_str = f"{date_str}_{time_str}"
                    created_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                else:
                    created_time = datetime.fromtimestamp(stat.st_mtime)

                recordings.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": created_time.isoformat(),
                    "duration": await self.get_video_duration(file_path),
                    "url": f"/recordings/{file_path.name}"
                })
            except Exception as e:
                print(f"[WEB] Error processing recording {file_path.name}: {e}")

        # Sort by creation time, newest first
        recordings.sort(key=lambda x: x["created"], reverse=True)

        return {"recordings": recordings}

    async def get_video_duration(self, file_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

                if fps > 0:
                    return frame_count / fps
            return 0.0
        except Exception:
            return 0.0

    async def serve_recording(self, request: Request, filename: str):
        """Serve a recording file with proper range request support for video streaming"""
        # Validate filename to prevent directory traversal
        if not filename.endswith('.mp4') or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        recordings_dir = Path("recordings")
        file_path = recordings_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Recording not found")

        # Get file size
        file_size = file_path.stat().st_size

        # Parse range header
        range_header = request.headers.get('range')

        if range_header:
            # Handle range request
            return await self._serve_video_range(file_path, range_header, file_size)
        else:
            # Serve entire file
            return await self._serve_video_full(file_path, file_size)

    async def _serve_video_range(self, file_path: Path, range_header: str, file_size: int):
        """Handle HTTP range requests for video streaming"""
        try:
            # Parse range header (e.g., "bytes=0-1023" or "bytes=1024-")
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            # Ensure valid range
            start = max(0, start)
            end = min(file_size - 1, end)
            content_length = end - start + 1

            def generate():
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)  # 8KB chunks
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': 'video/mp4',
            }

            return StreamingResponse(
                generate(),
                status_code=206,
                headers=headers,
                media_type='video/mp4'
            )

        except (ValueError, IndexError):
            # Invalid range header, serve full file
            return await self._serve_video_full(file_path, file_size)

    async def _serve_video_full(self, file_path: Path, file_size: int):
        """Serve the entire video file"""
        def generate():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk

        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Type': 'video/mp4',
        }

        return StreamingResponse(
            generate(),
            status_code=200,
            headers=headers,
            media_type='video/mp4'
        )

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
        .recordings {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .recording-item {
            display: flex;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
            gap: 15px;
        }
        .recording-item:last-child {
            border-bottom: none;
        }
        .recording-info {
            flex: 1;
        }
        .recording-title {
            font-weight: bold;
            color: #333;
        }
        .recording-meta {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .recording-controls {
            display: flex;
            gap: 10px;
        }
        .play-btn {
            background: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .play-btn:hover {
            background: #1976D2;
        }
        .download-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .download-btn:hover {
            background: #45a049;
        }
        #videoModal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
        }
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-width: 90%;
            max-height: 90%;
        }
        .close-btn {
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
        }
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

        <div class="recordings">
            <h2>📹 Alert Recordings</h2>
            <div id="recordingsList">
                <p>Loading recordings...</p>
            </div>
        </div>
    </div>

    <!-- Video Modal -->
    <div id="videoModal">
        <div class="modal-content">
            <button class="close-btn" onclick="closeVideoModal()">&times;</button>
            <video id="modalVideo" controls style="width: 100%; max-width: 800px;">
                Your browser does not support the video tag.
            </video>
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

        async function loadRecordings() {
            try {
                const response = await fetch("/recordings");
                if (response.ok) {
                    const data = await response.json();
                    displayRecordings(data.recordings);
                } else {
                    document.getElementById("recordingsList").innerHTML = "<p>Failed to load recordings</p>";
                }
            } catch (error) {
                console.error("Error loading recordings:", error);
                document.getElementById("recordingsList").innerHTML = "<p>Error loading recordings</p>";
            }
        }

        function displayRecordings(recordings) {
            const recordingsList = document.getElementById("recordingsList");

            if (recordings.length === 0) {
                recordingsList.innerHTML = "<p>No recordings available</p>";
                return;
            }

            const recordingsHtml = recordings.map(recording => {
                const date = new Date(recording.created);
                const formattedDate = date.toLocaleString();
                const fileSizeMB = (recording.size / (1024 * 1024)).toFixed(1);
                const durationStr = recording.duration ? `${recording.duration.toFixed(1)}s` : 'Unknown';

                return `
                    <div class="recording-item">
                        <div class="recording-info">
                            <div class="recording-title">${recording.filename}</div>
                            <div class="recording-meta">
                                📅 ${formattedDate} | ⏱️ ${durationStr} | 💾 ${fileSizeMB} MB
                            </div>
                        </div>
                        <div class="recording-controls">
                            <button class="play-btn" onclick="playRecording('${recording.url}', '${recording.filename}')">
                                ▶️ Play
                            </button>
                            <button class="download-btn" onclick="downloadRecording('${recording.url}', '${recording.filename}')">
                                ⬇️ Download
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            recordingsList.innerHTML = recordingsHtml;
        }

        function playRecording(url, filename) {
            const modal = document.getElementById("videoModal");
            const video = document.getElementById("modalVideo");

            video.src = url;
            modal.style.display = "block";

            // Add title to modal
            const existingTitle = modal.querySelector('.video-title');
            if (existingTitle) {
                existingTitle.remove();
            }

            const title = document.createElement('h3');
            title.className = 'video-title';
            title.textContent = filename;
            title.style.margin = '0 0 15px 0';

            const modalContent = modal.querySelector('.modal-content');
            modalContent.insertBefore(title, video);
        }

        function closeVideoModal() {
            const modal = document.getElementById("videoModal");
            const video = document.getElementById("modalVideo");

            modal.style.display = "none";
            video.pause();
            video.src = "";
        }

        function downloadRecording(url, filename) {
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById("videoModal");
            if (event.target === modal) {
                closeVideoModal();
            }
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeVideoModal();
            }
        });

        connectWebSocket();
        loadRecentEvents();
        loadRecordings();
    </script>
</body>
</html>
        '''