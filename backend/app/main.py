from fastapi import FastAPI, Depends

from app.api.people import router as people_router
from app.api.auth import router as auth_router

from app.models.user import User
from app.utils.auth_dependency import get_current_user


app = FastAPI(
    title="ThirdEye",
    description="AI-Based Face Sketch Recognition and Candidate Retrieval System",
    version="1.0.0"
)


# Include API routers
app.include_router(people_router)
app.include_router(auth_router)


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "ThirdEye Backend is running!",
        "status": "success"
    }


# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Protected current-user endpoint
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