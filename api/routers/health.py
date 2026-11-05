from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """Kubernetes-style liveness probe."""
    return {"status": "ok", "service": "predictive-maintenance-api"}
