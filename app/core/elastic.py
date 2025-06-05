from elasticsearch import Elasticsearch
from app.core.config import settings

es = Elasticsearch(
    settings.ELASTIC_URL,
    #api_key=settings.ELASTIC_API_KEY,
    # headers={
    #     "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
    #     "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"
    # }
)