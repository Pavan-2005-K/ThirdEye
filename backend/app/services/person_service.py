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


def update_person(
    db: Session,
    person_id: int,
    name: str | None,
    age: int | None,
    gender: str | None,
    address: str | None,
    phone: str | None,
    photo_path: str | None
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


def delete_person(db: Session, person_id: int):
    person = get_person_by_id(db, person_id)

    if person is None:
        return None

    db.delete(person)
    db.commit()

    return person