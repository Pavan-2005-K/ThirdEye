import os
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile
)

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.person_service import (
    create_person,
    get_all_people,
    get_person_by_id,
    update_person,
    delete_person
)


router = APIRouter(
    prefix="/people",
    tags=["Registered People"]
)


# Folder where registered-person photos will be stored
UPLOAD_DIR = "uploads/people"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


@router.get("/")
def list_people(
    db: Session = Depends(get_db)
):
    return get_all_people(db)


@router.get("/{person_id}")
def get_person(
    person_id: int,
    db: Session = Depends(get_db)
):
    person = get_person_by_id(db, person_id)

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Person not found"
        )

    return person


@router.post("/upload")
def add_person_with_photo(
    name: str = Form(...),
    photo: UploadFile = File(...),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    address: str | None = Form(None),
    phone: str | None = Form(None),
    db: Session = Depends(get_db)
):
    # Check file extension
    original_name = photo.filename or ""
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and WEBP images are allowed"
        )

    # Read uploaded file
    file_data = photo.file.read()

    if not file_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded photo is empty"
        )

    # Maximum 10 MB
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Photo size must be less than 10 MB"
        )

    # Create upload folder
    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    # Generate unique filename
    unique_name = f"{uuid.uuid4()}{extension}"

    photo_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    # Save photo
    with open(photo_path, "wb") as buffer:
        buffer.write(file_data)

    try:
        # Create database record and AI embedding
        person = create_person(
            db=db,
            name=name,
            age=age,
            gender=gender,
            address=address,
            phone=phone,
            photo_path=photo_path
        )

        return {
            "message": "Registered person added successfully",
            "person_id": person.id,
            "name": person.name,
            "photo_path": person.photo_path,
            "embedding_path": person.embedding_path
        }

    except Exception as e:
        # Remove uploaded photo if processing fails
        if os.path.exists(photo_path):
            os.remove(photo_path)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process person: {str(e)}"
        )


@router.post("/")
def add_person(
    name: str,
    photo_path: str,
    age: int | None = None,
    gender: str | None = None,
    address: str | None = None,
    phone: str | None = None,
    db: Session = Depends(get_db)
):
    return create_person(
        db=db,
        name=name,
        age=age,
        gender=gender,
        address=address,
        phone=phone,
        photo_path=photo_path
    )


@router.put("/{person_id}")
def edit_person(
    person_id: int,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    address: str | None = None,
    phone: str | None = None,
    photo_path: str | None = None,
    db: Session = Depends(get_db)
):
    person = update_person(
        db=db,
        person_id=person_id,
        name=name,
        age=age,
        gender=gender,
        address=address,
        phone=phone,
        photo_path=photo_path
    )

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Person not found"
        )

    return person


@router.delete("/{person_id}")
def remove_person(
    person_id: int,
    db: Session = Depends(get_db)
):
    person = delete_person(
        db,
        person_id
    )

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Person not found"
        )

    return {
        "message": "Person deleted successfully",
        "person_id": person_id
    }