from fastapi import FastAPI

from app.api.people import router as people_router
from app.api.auth import router as auth_router

app = FastAPI(
    title="ThirdEye",
    description="AI-Based Face Sketch Recognition and Candidate Retrieval System",
    version="1.0.0"
)


app.include_router(people_router)
app.include_router(auth_router)

@app.get("/")
def root():
    return {
        "message": "ThirdEye Backend is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }