import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from api.schema.schema import ApiSchema
from typing import Optional
from datetime import datetime

class TmdbSchema(ApiSchema):
    budget: float
    runtime: int
    # revenue: float
    release_date: datetime
    vote_average: float
    vote_count: int
    id: int
    imdb_id: str
    # original_language: str
    Genre_1: str
    Genre_2: Optional[str]
    Genre_3: Optional[str]
    popularity: float
    Production_Companies: Optional[str]
    Title: str