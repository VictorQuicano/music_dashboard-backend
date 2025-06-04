# Elasticsearch Video Search API

A FastAPI-based service for indexing and searching video metadata with Elasticsearch, featuring object detection capabilities.

## Features

- **Video Metadata Search**: Search across video attributes (filename, location, codec, resolution)
- **Object Detection Search**: Find videos containing specific detected objects
- **Video Upload**: Store videos in HDFS and index their metadata in Elasticsearch
- **Bulk Data Import**: Script for importing existing video datasets

## API Endpoints

### Search
- `GET /api/search?q={query}`
  - Searches across video metadata and detected objects
  - Returns matching videos with their full metadata

### Upload
- `POST /api/upload`
  - Accepts video file uploads
  - Stores video in HDFS
  - Returns generated video ID and storage path

## Project Structure

```
.
├── app/                      # FastAPI application core
│   ├── core/                 # Configuration and utilities
│   ├── models/               # Pydantic models
│   ├── routers/              # API endpoint definitions
│   └── main.py               # Application entry point
├── data/                     # Sample data files
├── bulk_data.py              # Bulk import script
├── docker-compose.yml        # Docker configuration
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
└── run.sh                    # Convenience script for local execution
```

## Setup

### Prerequisites
- Python 3.12+
- Docker
- Elasticsearch (via Docker)
- HDFS (optional, for video storage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/elastic_search_api.git
   cd elastic_search_api
   ```

2. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Start services:
   ```bash
   docker-compose up -d
   ```

5. Run the API:
   ```bash
   ./run.sh
   ```

## Usage

### Searching Videos
```bash
curl "http://localhost:8000/api/search?q=dog"
```

### Uploading Videos
```bash
curl -X POST -F "file=@your_video.mp4" http://localhost:8000/api/upload
```

### Bulk Import
```bash
python bulk_data.py --index videos --file data/sample_videos.json
```

## Development

- Format code:
  ```bash
  black .
  ```

- Run tests:
  ```bash
  pytest
  ```