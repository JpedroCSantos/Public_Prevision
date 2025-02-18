import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from api.schema.schema import ApiSchema
from typing import Optional

class OmdbSchema(ApiSchema):
    Director: Optional[str]
    Actors: Optional[str]
    Metascore: Optional[float]
    imdbRating: Optional[float]
    Rated: Optional[str]
    # BoxOffice: Optional[float]
