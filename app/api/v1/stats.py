from fastapi import APIRouter
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from app.db.session import SessionLocal
from app.db.models import ChatStats
from app.core.rate_limit import limiter
from fastapi import Request

router = APIRouter()

@router.get("/stats", summary="Get token usage and cost statistics by date")
@limiter.limit("30/minute")
def get_stats(request: Request):
    """
    Returns total token usage and estimated cost grouped by date.
    """
    db: Session = SessionLocal()
    try:
        # Group by date part of 'date' column
        stats_by_date = (
            db.query(
                cast(ChatStats.date, Date).label('usage_date'),
                func.sum(ChatStats.prompt_tokens).label('prompt_tokens'),
                func.sum(ChatStats.completion_tokens).label('completion_tokens'),
                func.sum(ChatStats.total_tokens).label('total_tokens'),
                func.sum(ChatStats.estimated_cost_usd).label('cost_usd')
            )
            .group_by(cast(ChatStats.date, Date))
            .order_by(cast(ChatStats.date, Date).desc())
            .all()
        )
        
        # Calculate overall totals
        total_tokens = sum(row.total_tokens for row in stats_by_date if row.total_tokens)
        total_cost = sum(row.cost_usd for row in stats_by_date if row.cost_usd)
        
        return {
            "overall_total_tokens": total_tokens,
            "overall_estimated_cost_usd": round(total_cost, 4),
            "usage_by_date": [
                {
                    "date": row.usage_date.isoformat(),
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "total_tokens": row.total_tokens,
                    "estimated_cost_usd": round(row.cost_usd, 4)
                }
                for row in stats_by_date
            ]
        }
    finally:
        db.close()
