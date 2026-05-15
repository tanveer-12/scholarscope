# src/pipeline/resources/database.py
# Dagster resource = shared dependency injected into assets.
# Think of it like a singleton database connection
# that every asset can use without creating its own.

from dagster import ConfigurableResource
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

class DatabaseResource(ConfigurableResource):
    postgres_url: str

    def get_session(self):
        engine = create_engine(self.postgres_url)
        Session = sessionmaker(bind=engine)
        return Session()
    
    def get_engine(self):
        return create_engine(self.postgres_url)