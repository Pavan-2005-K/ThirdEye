import os
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile
)

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.user import User
from app.models.investigation import Investigation
from app.utils.auth_dependency import get_current_user


router = APIRouter(
    prefix="/sketch",
    tags=["Sketch"]
)


UPLOAD_DIR = "uploads/sketches"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


@router.post("/upload")
def upload_sketch(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check file extension
    original_name = file.filename or ""
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and WEBP images are allowed"
        )

    # Read file
    file_data = file.file.read()

    if not file_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Limit file size to 10 MB
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size must be less than 10 MB"
        )

    # Generate unique filename
    unique_name = f"{uuid.uuid4()}{extension}"

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(file_data)

    # Create investigation
    investigation = Investigation(
        user_id=current_user.id,
        sketch_path=file_path
    )

    db.add(investigation)
    db.commit()
    db.refresh(investigation)

    return {
        "message": "Sketch uploaded successfully",
        "investigation_id": investigation.id,
        "user_id": current_user.id,
        "filename": unique_name,
        "sketch_path": file_path
    }