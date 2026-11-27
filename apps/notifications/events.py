"""Event types pushed over webhooks and WebSockets."""

MEDIA_UPLOADED = "media.uploaded"
JOB_COMPLETED = "job.completed"
JOB_FAILED = "job.failed"
JOB_CANCELLED = "job.cancelled"
DETECTION_TRIGGERED = "alert.detection_triggered"
PING = "ping"

EVENT_TYPES = (MEDIA_UPLOADED, JOB_COMPLETED, JOB_FAILED, JOB_CANCELLED, DETECTION_TRIGGERED, PING)
ALL = "*"


def user_group(owner_id) -> str:
    """Channels group for one user's alerts."""
    return f"user-{owner_id}"


def flight_group(owner_id, flight_id) -> str:
    """Channels group for one user's flight telemetry (flight ids are owner-scoped)."""
    return f"flight-{owner_id}-{flight_id}"
