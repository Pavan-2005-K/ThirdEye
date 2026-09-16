from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.person_service import (
    create_person,
    get_all_people,
    get_person_by_id
)

router = APIRouter(
    prefix="/people",
    tags=["Registered People"]
)


@router.get("/")
def list_people(db: Session = Depends(get_db)):
    return get_all_people(db)


@router.get("/{person_id}")
def get_person(
    person_id: int,
    db: Session = Depends(get_db)
):
    return get_person_by_id(db, person_id)


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