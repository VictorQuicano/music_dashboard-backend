from elasticsearch import Elasticsearch
from app.core.config import settings

es = Elasticsearch(
    settings.ELASTIC_URL,
    api_key=(settings.ELASTIC_API_KEY if settings.ELASTIC_API_KEY else None),
)