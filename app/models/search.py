from pydantic import BaseModel

class SearchResponse(BaseModel):
    hits: list