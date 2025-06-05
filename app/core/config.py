import os
from dotenv import load_dotenv

load_dotenv() 

class Settings:
    APP_ENVIROMENT = os.getenv("APP_ENVIROMENT", "LOCAL")
    ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
    ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")

    TMP_DIR = os.getenv("TMP_DIR", "/tmp")

    if APP_ENVIROMENT == "LOCAL":
        MUSIC_4_ALL_PATH = os.getenv("MUSIC_4_ALL_PATH") 
        MUSIC_4_ALL_ONION_PATH = os.getenv("MUSIC_4_ALL_ONION_PATH")
        EXTRASENSORY = os.getenv("EXTRASENSORY") 
        MUSIC_REGISTERS = f"{MUSIC_4_ALL_ONION_PATH}/timestamps"
        MUSIC_EXTRASENSORY = os.getenv("MUSIC_EXTRASENSORY")

settings = Settings()
