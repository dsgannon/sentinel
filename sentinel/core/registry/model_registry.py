from sqlalchemy import create_engine, Column, String, Integer, Date, text
from sqlalchemy.orm import declarative_base, Session
import os

Base = declarative_base()

def get_engine():
    """
    Create a SQLAlchemy engine from the DATABASE_URL environment variable.

    Defaults to a local SQLite database for development. Set DATABASE_URL
    to a PostgreSQL connection string for production (e.g. AWS RDS).

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    url = os.getenv("DATABASE_URL", "sqlite:///sentinel_registry.db")
    return create_engine(url)

class ModelRecord(Base):
    __tablename__ = "models"

    model_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    owner = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    purpose = Column(String)
    training_date = Column(String)
    validation_status = Column(String, default="pending")
    next_review_date = Column(String)
    tier = Column(Integer)

def init_db():
    """
    Create all registry tables if they do not already exist.

    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS
    semantics. Must be called before any other registry operations.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)

def register_model(model_id, model_name, owner, model_type, **kwargs):
    """
    Insert a new model record into the registry.

    Parameters
    ----------
    model_id : str
        Unique identifier for the model (e.g. "credit_risk_v2").
    model_name : str
        Human-readable model name.
    owner : str
        Name of the model owner or responsible team.
    model_type : str
        Model type (e.g. "classification", "regression").
    **kwargs
        Optional fields: purpose, training_date, validation_status,
        next_review_date, tier.
    """
    engine = get_engine()
    with Session(engine) as session:
        record = ModelRecord(
            model_id=model_id,
            model_name=model_name,
            owner=owner,
            model_type=model_type,
            **kwargs
        )
        session.add(record)
        session.commit()


def get_model(model_id):
    """
    Retrieve a model record from the registry by ID.

    Parameters
    ----------
    model_id : str
        Unique identifier for the model.

    Returns
    -------
    ModelRecord or None
        The model record if found, None if not registered.
    """
    engine = get_engine()
    with Session(engine) as session:
        return session.get(ModelRecord, model_id)
    