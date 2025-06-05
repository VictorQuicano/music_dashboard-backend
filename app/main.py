from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.upload import router as upload_router
from app.api.search import router as search_router

from app.api.endpoints.music import router as music_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#index_music_registers()

app.include_router(music_router, prefix="/api")

app.include_router(upload_router, prefix="/api")
app.include_router(search_router, prefix="/api")

@app.get("/api/hello")
def read_hello():
    return {"message": "Hello from FastAPI!"}