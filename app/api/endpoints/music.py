# app/api/endpoints/music.py
import os
import json
from typing import Dict

from fastapi import FastAPI, HTTPException, Query, APIRouter
from elasticsearch import Elasticsearch, helpers
from app.core.elastic import es
from app.core.config import settings
import pandas as pd

router = APIRouter()

@router.get("/songs_count")
def songs_count() -> Dict[str, int]:
    """
    Returns the total number of songs indexed in Elasticsearch.
    """
    try:
        res = es.count(index="song")
        return {"total_songs": res["count"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/languages")
def languages():
    try:
        path = os.path.join(
            '/media/alejandro/Extra/Carrera/5th_Year/IX_SEMESTRER/TCD/1er_parcial/Models/output',
            'lang_compact.json'
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/songs_global")
def songs_global():
    res = es.search(index="music", size=0, aggs={
        "songs_per_year": {
            "terms": {
                "field": "year",
                "size": 100,
                "order": {"_key": "asc"}
            }
        }
    })
    return {
        "years": {bucket["key"]: bucket["doc_count"] for bucket in res["aggregations"]["songs_per_year"]["buckets"]}
    }
"""
lang_df = pd.read_csv(os.path.join(settings.MUSIC_4_ALL_PATH, "id_lang.csv"))



@router.get("/idiomas_info")
def top_idiomas():
      
    

@router.get("/songs_query")
def songs_query(year: int = Query(...), month: int = Query(...)):
    res = es.search(index="music", size=1000, query={
        "bool": {
            "must": [
                {"term": {"year": year}},
                {"term": {"month": month}}
            ]
        }
    })
    hits = [hit["_source"] for hit in res["hits"]["hits"]]
    return {"results": hits}
"""
