from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.matching_service import search_all_datasets
from app.models.user import User
from app.models.investigation import Investigation
from app.utils.auth_dependency import get_current_user


router = APIRouter(
    prefix="/matching",
    tags=["Matching"]
)


@router.post("/search")
def search_matches(
    investigation_id: int,
    top_k: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find the investigation
    investigation = (
        db.query(Investigation)
        .filter(
            Investigation.id == investigation_id,
            Investigation.user_id == current_user.id
        )
        .first()
    )

    if not investigation:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found"
        )

    try:
        results = search_all_datasets(
            db=db,
            sketch_path=investigation.sketch_path,
            top_k=top_k
        )

        # Save top match into investigation history
        if results:
            top_match = results[0]

            if top_match.get("source") == "admin_dataset":
                investigation.top_match_person_id = top_match.get("person_id")

            investigation.top_similarity = top_match.get("similarity")

            db.commit()

        return {
            "message": "Matching completed successfully",
            "investigation_id": investigation.id,
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