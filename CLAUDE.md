# Doodie Duty - AI Assistant Guide

## Project Overview
Doodie Duty is a computer vision application that detects when dogs are left unsupervised and triggers customizable alerts. It uses YOLOv8 for object detection and provides a web interface for monitoring.

## Core Purpose
- Detect dogs and humans in camera feed
- Determine if dogs are unsupervised (no humans present)
- Trigger alerts after configurable delay
- Log events to database
- Provide real-time web interface

## Technical Stack
- **Python 3.13** - Core language
- **YOLOv8** - Object detection (dogs and humans)
- **OpenCV** - Camera capture and image processing
- **FastAPI** - Web framework with WebSocket support
- **SQLAlchemy + SQLite** - Event storage
- **Pydantic** - Configuration management

## Project Structure
```
doodie_duty/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── .env                # Environment configuration (create from .env.example)
├── src/
│   ├── camera.py       # Camera capture (sync/async)
│   ├── detector.py     # YOLOv8 dog/human detection
│   ├── supervisor.py   # Supervision logic & state management
│   ├── web_app.py      # FastAPI web interface
│   ├── database.py     # Event logging & storage
│   ├── actions.py      # Alert actions (sound, video, etc.)
│   └── config.py       # Configuration management
├── test_app.py         # Component tests
├── logs/               # Event logs (auto-created)
├── recordings/         # Alert videos (auto-created)
└── doodie_duty.db     # SQLite database (auto-created)
```

## Key Components

### Detection System
- Uses YOLOv8n model (lightweight, downloads automatically)
- Class IDs: Dog=16, Person=0
- Confidence threshold: 0.5 (configurable)

### Supervision States
1. **IDLE** - No dogs detected
2. **SUPERVISED** - Dogs with humans present
3. **UNSUPERVISED** - Dogs without humans
4. **ALERT** - Unsupervised beyond threshold

### Action Triggers
- **Sound Alert** - System TTS or custom sound file
- **Video Recording** - 30-second clips on alert
- **File Logging** - JSON logs to disk
- **Webhook** - HTTP notifications

## Development Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Application
```bash
python main.py
# With options:
python main.py --camera 0 --port 8000 --config config.json
```

### Run Tests
```bash
python test_app.py
```

### Lint/Type Check Commands
```bash
# Add these if linting tools are set up:
# ruff check src/
# mypy src/
```

## Configuration

### Key Settings (.env)
- `CAMERA_INDEX=0` - Camera device (0=default, 1=USB)
- `ALERT_DELAY_SECONDS=5` - Time before alert triggers
- `PORT=8000` - Web server port
- `ENABLE_SOUND_ALERT=true` - Toggle sound alerts
- `ENABLE_VIDEO_RECORDING=true` - Toggle recording

### Raspberry Pi Optimizations
- `USE_LIGHTWEIGHT_MODEL=true` - Use smaller model
- `REDUCE_RESOLUTION=true` - Lower camera resolution
- `TARGET_WIDTH=640` - Reduced frame width
- `TARGET_HEIGHT=480` - Reduced frame height

## Common Tasks

### Add New Action Trigger
1. Create new class in `src/actions.py` extending `ActionTrigger`
2. Implement `_execute()` method
3. Add to `ActionManager` in `main.py`

### Modify Detection Classes
1. Edit `DOG_CLASSES` and `HUMAN_CLASSES` in `src/detector.py`
2. YOLOv8 class names are in `self.model.names`

### Change Alert Timing
1. Modify `ALERT_DELAY_SECONDS` in `.env`
2. Or pass to `DogSupervisor` constructor

## API Endpoints
- `GET /` - Web interface HTML
- `GET /status` - Current system status (JSON)
- `GET /events?limit=10` - Recent events (JSON)
- `POST /start` - Start monitoring
- `POST /stop` - Stop monitoring
- `WebSocket /ws` - Real-time updates & video stream

## WebSocket Messages

### Client → Server
```json
{"type": "get_frame"}     // Request video frame
{"type": "get_status"}    // Request current status
```

### Server → Client
```json
{"type": "frame", "data": {...}}   // Video frame + detections
{"type": "event", "data": {...}}   // Supervision event
{"type": "status", "data": {...}}  // System status
```

## Known Issues & Considerations

### Camera Access
- macOS requires camera permissions
- Default camera index is 0, USB cameras typically 1 or 2
- Camera must support 640x480 minimum resolution

### Model Download
- YOLOv8n.pt (6.25MB) downloads on first run
- Requires internet connection for initial setup
- Model cached locally after download

### Performance
- Detection runs at ~2 FPS on Raspberry Pi 4
- Reduce FPS and resolution for better Pi performance
- Consider using threading for camera capture

### Database
- SQLite may have issues with concurrent writes
- Consider PostgreSQL for production deployment
- Events older than 30 days auto-deleted

## Deployment Notes

### Raspberry Pi
1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-opencv libatlas-base-dev
   ```
2. Use lightweight settings in `.env`
3. Consider USB camera over Pi camera module
4. Ensure 3A power supply

### Docker (Future)
- Not yet implemented
- Would need to handle camera device access
- Consider using `--device /dev/video0` flag

### Security
- No authentication on web interface yet
- Run behind reverse proxy (nginx) for production
- Sanitize webhook URLs before deployment
- Don't expose port 8000 directly to internet

## Testing Approach
- Unit tests for individual components in `test_app.py`
- Mock camera for CI/CD environments
- Test detection with sample images
- Verify database operations independently

## Future Enhancements
- User authentication for web interface
- Multiple camera support
- Custom training for specific dog breeds
- Mobile app notifications
- Cloud storage for recordings
- Historical analytics dashboard
- Email alerts
- Zone-based detection (specific areas)

## Debugging Tips
- Check camera with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- View YOLOv8 classes: `python -c "from ultralytics import YOLO; print(YOLO('yolov8n.pt').names)"`
- Test detection on image: Use `detector.detect()` with loaded image
- Database issues: Delete `doodie_duty.db` and restart
- WebSocket issues: Check browser console for errors

## Contact & Support
This application was developed as a personal project for dog supervision monitoring.
For issues or questions about the codebase, refer to the inline documentation and comments.