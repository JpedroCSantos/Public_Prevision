import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from api.schema.schema import ApiSchema
from typing import Optional
from datetime import datetime

class TmdbSchema(ApiSchema):
    budget: Optional[float]
    runtime: Optional[int]
    # revenue: float
    release_date: Optional[datetime]
    vote_average: Optional[float]
    vote_count: Optional[int]
    id: Optional[int]
    imdb_id: Optional[str]
    # original_language: str
    Genre_1: Optional[str]
    Genre_2: Optional[str]
    Genre_3: Optional[str]
    popularity: Optional[float]
    Production_Companies: Optional[str]
    Title: Optional[str]
    belongs_to_collection: Optional[int]