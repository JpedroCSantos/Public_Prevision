from typing import Optional
from pydantic import BaseModel


class MovieSchema(BaseModel):
    id_movie_tmdb: int
    Year: int
    Title: str
    WorldwideBox_office: float
    DomesticBox_office: float
    InternationalBox_office: float
    production_cost: float
    Runtime: int
    release_date: int
    Genre_1: str
    Genre_2: Optional[str]
    Genre_3: Optional[str]
    original_language: str
    popularity: Optional[float]
    Production_Companies: Optional[str]
    vote_average: Optional[float]
    vote_count: Optional[int]
    Cast_1: str
    Cast_2: Optional[str]
    Cast_3: Optional[str]
    Director_1: str
    IMDB_Rating: float
    Metascore: Optional[float]

    class Config:
        from_attributes: True
