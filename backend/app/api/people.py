from fastapi import APIRouter, Depends, HTTPException
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


@router.get("/")
def list_people(db: Session = Depends(get_db)):
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
    person = delete_person(db, person_id)

    if person is None:
        raise HTTPException(
            status_code=404,
            detail="Person not found"
        )

    return {
        "message": "Person deleted successfully",
        "person_id": person_id
    }