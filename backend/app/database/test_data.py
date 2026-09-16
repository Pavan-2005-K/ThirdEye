from app.database.database import SessionLocal
from app.models.user import User
from app.models.person import Person
from app.models.investigation import Investigation


db = SessionLocal()

try:
    # Create a test user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash="test_hash"
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    print(f"User created: ID = {user.id}")

    # Create a test registered person
    person = Person(
        name="Test Person",
        age=30,
        gender="Unknown",
        address="Test Address",
        phone="0000000000",
        photo_path="test/person.jpg"
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    print(f"Registered person created: ID = {person.id}")

    # Create an investigation
    investigation = Investigation(
        user_id=user.id,
        sketch_path="test/sketch.jpg",
        top_match_person_id=person.id,
        top_similarity=85.5
    )

    db.add(investigation)
    db.commit()
    db.refresh(investigation)

    print(f"Investigation created: ID = {investigation.id}")

    print("\nDatabase test successful!")

except Exception as e:
    db.rollback()
    print("Database test failed!")
    print(e)

finally:
    db.close()