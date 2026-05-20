from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from recommender import AnimeRecommender
import os

app = FastAPI(title="Anime Recommender API")

# Initialize recommender engine (loads all models on startup)
recommender = AnimeRecommender(data_path="anime.csv", cf_path="cf_model.json", xgb_path="xgb_model.json")

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/api/search")
def search_anime(q: str = Query(..., min_length=1), limit: int = 10):
    """Search for anime by title"""
    return recommender.search_anime(q, limit)

@app.get("/api/recommend")
def get_recommendations(
    title: str = Query(...), 
    model: str = Query("hybrid"),
    limit: int = 10, 
    sort_by_rating: bool = False
):
    """Get recommendations based on an anime title and selected model"""
    return recommender.get_recommendations(title, model, limit, sort_by_rating)

@app.get("/api/stats")
def get_stats():
    """Get basic stats about the dataset"""
    return recommender.get_stats()
