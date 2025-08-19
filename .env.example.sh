# Example environment variables
# Copy this file to .env and fill in your values

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-change-this-in-production
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# File Upload Configuration
MAX_CONTENT_LENGTH=524288000  # 500MB in bytes
UPLOAD_FOLDER=uploads
PROCESSED_FOLDER=processed

# Processing Configuration
DEFAULT_DISTANCE_THRESHOLD=200
DEFAULT_DURATION_THRESHOLD=600
DEFAULT_PROBABILITY_THRESHOLD=0.1
DEFAULT_SPEED_THRESHOLD_KMH=120