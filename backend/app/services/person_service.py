from sqlalchemy.orm import Session

from app.models.person import Person
from app.ai.person_siamese_embedding import create_siamese_person_embedding


def create_person(
    db: Session,
    name: str,
    age: int | None,
    gender: str | None,
    address: str | None,
    phone: str | None,
    photo_path: str
):
    person = Person(
        name=name,
        age=age,
        gender=gender,
        address=address,
        phone=phone,
        photo_path=photo_path
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    # Generate V2 AI embedding after the person gets an ID
    embedding_path = create_siamese_person_embedding(
        person_id=person.id,
        photo_path=photo_path
    )

    person.embedding_path = embedding_path

    db.commit()
    db.refresh(person)

    return person


def get_all_people(db: Session):
    return db.query(Person).all()


def get_person_by_id(
    db: Session,
    person_id: int
):
    return (
        db.query(Person)
        .filter(Person.id == person_id)
        .first()
    )


def update_person(
    db: Session,
    person_id: int,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    address: str | None = None,
    phone: str | None = None,
    photo_path: str | None = None
):
    person = get_person_by_id(db, person_id)

    if person is None:
        return None

    if name is not None:
        person.name = name

    if age is not None:
        person.age = age

    if gender is not None:
        person.gender = gender

    if address is not None:
        person.address = address

    if phone is not None:
        person.phone = phone

    if photo_path is not None:
        person.photo_path = photo_path

    db.commit()
    db.refresh(person)

    return person


def delete_person(
    db: Session,
    person_id: int
):
    person = get_person_by_id(db, person_id)

    if person is None:
        return None

    db.delete(person)
    db.commit()

    return person