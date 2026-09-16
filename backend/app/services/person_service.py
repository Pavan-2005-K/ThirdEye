from sqlalchemy.orm import Session

from app.models.person import Person


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

    return person


def get_all_people(db: Session):
    return db.query(Person).all()


def get_person_by_id(db: Session, person_id: int):
    return db.query(Person).filter(Person.id == person_id).first()