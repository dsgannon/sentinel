from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime
import uuid
from sentinel.core.registry.model_registry import Base, get_engine

class AuditEntry(Base):
    __tablename__ = "audit_log"

    entry_id = Column(String, primary_key = True)
    timestamp = Column(DateTime, nullable = False)
    user = Column(String, nullable = False)
    model_id = Column(String, nullable = False)
    action = Column(String, nullable = False)
    details = Column(JSON)

def log_action(user, model_id, action, details = None):
    """
    Append an immutable audit entry to the validation log.

    Entries are insert-only — no update or delete operations are
    exposed, ensuring an tamper-evident record of all validation
    actions for regulatory examination.

    Parameters
    ----------
    user : str
        Name or ID of the user performing the action.
    model_id : str
        ID of the model being acted upon.
    action : str
        Description of the action (e.g. "validation_run",
        "status_change", "tier_assignment").
    details : dict, optional
        Additional context as a JSON-serializable dict
        (e.g. {"old_status": "pending", "new_status": "approved"}).
    """
    engine = get_engine()
    with Session(engine) as session:
        entry = AuditEntry(
            entry_id = str(uuid.uuid4()),
            timestamp = datetime.utcnow(),
            user = user,
            model_id = model_id,
            action = action,
            details = details
        )
        session.add(entry)
        session.commit()