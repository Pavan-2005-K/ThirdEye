from app.database.database import engine, Base
from app.models.person import Person
from app.models.user import User
from app.models.investigation import Investigation

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")