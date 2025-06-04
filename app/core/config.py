import os
from dotenv import load_dotenv

load_dotenv() 

class Settings:
    ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
    ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")

    HDFS_URL = os.getenv("HDFS_URL", "http://localhost:9870")
    HDFS_USER = os.getenv("HDFS_USER", "hadoop")

    TMP_DIR = os.getenv("TMP_DIR", "/tmp")
    HDFS_RAW_VIDEO_PATH = os.getenv("HDFS_RAW_VIDEO_PATH", "/videos/raw")

settings = Settings()
