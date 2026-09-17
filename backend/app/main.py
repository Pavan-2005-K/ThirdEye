from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles

from app.api.people import router as people_router
from app.api.auth import router as auth_router
from app.api.sketch import router as sketch_router
from app.api.matching import router as matching_router

from app.models.user import User
from app.utils.auth_dependency import get_current_user


app = FastAPI(
    title="ThirdEye",
    description="AI-Based Face Sketch Recognition and Candidate Retrieval System",
    version="1.0.0"
)


# =========================================================
# STATIC FILES
# =========================================================

# Uploaded files
app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


# Old dataset images
app.mount(
    "/dataset-images",
    StaticFiles(directory="data/online_dataset/images"),
    name="dataset-images"
)


# FS2K dataset photos
app.mount(
    "/fs2k-images",
    StaticFiles(directory="data/online_dataset/FS2K/photo"),
    name="fs2k-images"
)


# =========================================================
# API ROUTERS
# =========================================================

app.include_router(people_router)
app.include_router(auth_router)
app.include_router(sketch_router)
app.include_router(matching_router)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "ThirdEye Backend is running!",
        "status": "success"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# =========================================================
# CURRENT USER
# =========================================================

@app.get("/me")
def get_my_profile(
    current_user: User = Depends(get_current_user)
):
    return {
        "message": "Authenticated user",
        "user_id": current_user.id,
        "name": current_user.name,
        "email": current_user.email
    }