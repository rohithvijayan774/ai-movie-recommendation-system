from fastapi import FastAPI
from recommender import recommend

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Movie Recommendation API Running"}

@app.get("/recommend")
def get_recommendation(movie: str):
    return {"recommendations": recommend(movie)}
