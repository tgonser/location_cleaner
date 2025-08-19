# 🔍 Location Data Processor

A web-based tool for processing and filtering Google Location History data with real-time diagnostics and smart recommendations.

## ✨ Features

- **Real-time Processing**: Watch your data being processed with live diagnostic logs
- **Smart Filtering**: Filter activities by distance, visits by duration and probability
- **Actionable Insights**: Get specific recommendations to improve data retention
- **Date Range Filtering**: Process only the data you need
- **Clean Web Interface**: Modern, responsive design with progress tracking
- **Meaningful Downloads**: Output files named with your date ranges

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd location-data-processor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:5000`

## 📊 How It Works

### Input
- **Google Location History JSON** files exported from Google Takeout
- **Date range** for filtering
- **Threshold settings** for activities and visits

### Processing Pipeline
1. **Load** your JSON file
2. **Filter by date range** to focus on relevant data
3. **Process activities** (filter by distance threshold)
4. **Process visits** (filter by duration and probability)
5. **Generate timeline paths** from position data
6. **Output clean JSON** ready for analysis

### Output
- Filtered JSON file with activities, visits, and timeline paths
- Processing statistics and recommendations
- Meaningful filename: `location_data_2024-01-01_to_2024-12-31.json`

## ⚙️ Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| **Distance Threshold** | 200m | Minimum distance for activities to be kept |
| **Duration Threshold** | 600s | Minimum duration for visits to be kept |
| **Probability Threshold** | 0.1 | Minimum probability for visits to be kept |
| **Speed Threshold** | 120 km/h | Maximum speed for position filtering |

## 🧠 Smart Recommendations

The tool automatically suggests adjustments when data retention is low:

- **Low Activity Retention** (<30%): "Try distance_threshold = 50m"
- **Low Visit Retention** (<30%): "Try duration_threshold = 300s" or "Try probability_threshold = 0.05"
- **No Data Found**: "Try expanding the date range"

## 📁 Project Structure

```
location-data-processor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload storage (auto-created)
├── processed/            # Output files (auto-created)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔒 Security Notes

- No API keys or credentials required
- All processing happens locally
- Upload files are automatically cleaned up after processing
- No data is stored permanently on the server

## 🐛 Troubleshooting

### Common Issues

**"No entries found in date range"**
- Check that your date range overlaps with your location data
- Try expanding the date range

**Low retention rates**
- Follow the smart recommendations in the diagnostic log
- Lower thresholds to keep more data
- Check if your location data has the expected activity types

**File upload fails**
- Ensure you're uploading a valid JSON file
- Check file size (max 500MB)
- Verify the file is from Google Location History export

## 📝 Example Processing Log

```
📊 Loaded 70,626 total entries
🎯 Target date range: 2024-01-01 to 2024-12-31
📅 Entries in date range: 15,234
🚗 Activities: 1,200/2,500 kept (48.0%)
💡 LOW ACTIVITY RETENTION: Try lowering distance_threshold to 50m
🏠 Visits: 800/1,800 kept (44.4%)
💡 Many visits filtered by duration - consider duration_threshold = 300s
✅ Processing completed successfully!
📥 Download ready: location_data_2024-01-01_to_2024-12-31.json
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please ensure you comply with Google's terms of service when using location data.

## ⚠️ Privacy

- This tool processes your personal location data locally
- No data is sent to external servers
- All processing happens on your machine
- Delete uploaded files after processing if desired