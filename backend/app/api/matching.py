from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.matching_service import search_registered_people
from app.models.user import User
from app.utils.auth_dependency import get_current_user


router = APIRouter(
    prefix="/matching",
    tags=["Matching"]
)


@router.post("/search")
def search_matches(
    sketch_path: str,
    top_k: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        results = search_registered_people(
            db=db,
            sketch_path=sketch_path,
            top_k=top_k
        )

        return {
            "message": "Matching completed successfully",
            "searched_by_user": current_user.id,
            "total_matches": len(results),
            "matches": results
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )