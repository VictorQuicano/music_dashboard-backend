from fastapi import APIRouter, Query
from app.core.elastic import es
from app.models.search import SearchResponse

router = APIRouter()

@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1)):
    res = es.search(index="videos", query={
        "bool": {
            "should": [
                { "multi_match": {
                    "query": q,
                    "fields": ["filename", "location", "codec", "resolution"]
                }},
                { "nested": {
                    "path": "objects_detected",
                    "query": {
                        "match": {
                            "objects_detected.type": q
                        }
                    }
                }}
            ]
        }
    })
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return {"hits": hits}