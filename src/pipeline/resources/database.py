# src/pipeline/resources/database.py
# Dagster resource = shared dependency injected into assets.
# Think of it like a singleton database connection
# that every asset can use without creating its own.

import os
from dagster import ConfigurableResource
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import structlog

logger = structlog.get_logger()

class DatabaseResource(ConfigurableResource):
    postgres_url: str

    def get_engine(self):
        """
        returns a SQLAlchemy engine with connection pooling.
        QueuePool maintains up to 10 connections and recycles idle connections after 30 minutes
        """
        return create_engine(
            self.postgres_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
            echo=False,
        )
    
    def get_session(self):
        """Returns a new database session."""
        engine = self.get_engine()
        Session = sessionmaker(bind=engine)
        return Session()
    
    def execute(self, sql:str, params:dict = None):
        """
        Execute a single SQL statement and return results
        Useful for quick queries without managing sessions
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result