# Doodie Duty - Dog Supervision Monitor 🐕

An intelligent application that detects when a dog is in an area unsupervised and triggers customizable actions.

## Features

- **Real-time Detection**: Uses YOLOv8 to detect dogs and humans in camera feed
- **Supervision Monitoring**: Determines if dogs are supervised or alone
- **Web Interface**: Live camera feed with detection overlay
- **Alert System**: Configurable alerts when dogs are unsupervised
- **Action Triggers**:
  - Sound alerts
  - Video recording
  - File logging
  - Webhook notifications
- **Event History**: Database storage of all events with statistics
- **Cross-platform**: Works on macOS, Linux, and Raspberry Pi

## Installation

1. **Clone the repository** (or create the files as shown):
```bash
cd doodie_duty
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the application with default settings:
```bash
python main.py
```

Open your browser to `http://localhost:8000` to view the web interface.

## Configuration

### Command Line Options
```bash
python main.py --camera 0 --port 8000 --config config.json
```

### Environment Variables (.env file)
Create a `.env` file to customize settings:
```env
# Camera
CAMERA_INDEX=0
CAMERA_FPS=30

# Detection
MODEL_NAME=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5

# Alerts
ALERT_DELAY_SECONDS=5
CHECK_INTERVAL_SECONDS=0.5

# Web Server
HOST=0.0.0.0
PORT=8000

# Actions
ENABLE_SOUND_ALERT=true
ENABLE_VIDEO_RECORDING=true
RECORDING_DURATION=30

# Raspberry Pi Optimization
USE_LIGHTWEIGHT_MODEL=false
REDUCE_RESOLUTION=false
```

## Usage

### Web Interface
1. Navigate to `http://localhost:8000`
2. Click "Start Monitoring" to begin detection
3. View real-time camera feed with detection boxes
4. Monitor the state (Idle, Supervised, Unsupervised, Alert)
5. Review event history at the bottom of the page

### Detection States
- **Idle**: No dogs detected
- **Supervised**: Dogs detected with humans present
- **Unsupervised**: Dogs detected without humans
- **Alert**: Dogs unsupervised for longer than alert delay

## Raspberry Pi Deployment

### Optimization for Pi
1. Enable lightweight mode in `.env`:
```env
USE_LIGHTWEIGHT_MODEL=true
REDUCE_RESOLUTION=true
TARGET_WIDTH=640
TARGET_HEIGHT=480
```

2. Install Pi-specific dependencies:
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y python3-opencv libatlas-base-dev

# Use Pi camera if available
CAMERA_INDEX=0  # Or 1 for USB camera
```

3. Run with reduced resources:
```bash
python main.py --config pi_config.json
```

### Remote Access
To access from other devices on your network:
```bash
# Find your Pi's IP address
hostname -I

# Run with network-accessible host
python main.py
```

Access from any device: `http://[PI_IP_ADDRESS]:8000`

## Project Structure
```
doodie_duty/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── .env                # Environment configuration
├── src/
│   ├── camera.py       # Camera capture module
│   ├── detector.py     # Dog/human detection
│   ├── supervisor.py   # Supervision logic
│   ├── web_app.py      # Web interface
│   ├── database.py     # Event storage
│   ├── actions.py      # Alert actions
│   └── config.py       # Configuration management
├── logs/               # Event logs (created automatically)
├── recordings/         # Video recordings (created automatically)
└── doodie_duty.db     # SQLite database (created automatically)
```

## Troubleshooting

### Camera Not Found
- Check camera index: Try 0, 1, or 2
- Verify camera permissions
- Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Model Download
The YOLOv8 model will be downloaded automatically on first run. Ensure internet connection.

### Performance Issues
- Reduce camera FPS: `CAMERA_FPS=15`
- Increase check interval: `CHECK_INTERVAL_SECONDS=1.0`
- Use lightweight model: `USE_LIGHTWEIGHT_MODEL=true`

### Raspberry Pi Specific
- Ensure sufficient power supply (3A recommended)
- Use heatsinks for better performance
- Consider USB camera over Pi camera module for better compatibility

## Development

### Adding Custom Actions
Create new action triggers by extending `ActionTrigger` in `src/actions.py`:
```python
class CustomAction(ActionTrigger):
    async def _execute(self, event_data):
        # Your custom action logic
        return True
```

### API Endpoints
- `GET /` - Web interface
- `GET /status` - Current system status
- `GET /events` - Recent events
- `POST /start` - Start monitoring
- `POST /stop` - Stop monitoring
- `WebSocket /ws` - Real-time updates

## License

MIT

## Contributing

Feel free to submit issues and enhancement requests!