from db.db import Base
from sqlalchemy import Column, DateTime, Float, Integer, Numeric, String
from sqlalchemy.sql import func


class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True)
    id_movie_tmdb = Column(Integer)
    Year = Column(Integer, nullable=False)
    Title = Column(String, nullable=False)
    WorldwideBox_office = Column(Numeric(precision=11, scale=2), nullable=False)
    DomesticBox_office = Column(Numeric(precision=11, scale=2), nullable=False)
    InternationalBox_office = Column(Numeric(precision=11, scale=2), nullable=False)
    production_cost = Column(Numeric(precision=11, scale=2), nullable=False)
    Runtime = Column(Integer)
    release_date = Column(Integer)
    Genre_1 = Column(String, nullable=False)
    Genre_2 = Column(String)
    Genre_3 = Column(String)
    original_language = Column(String)
    popularity = Column(Integer)
    production_companies = Column(String)
    vote_average = Column(Integer)
    vote_count = Column(Integer)
    Cast_1 = Column(String)
    Cast_2 = Column(String)
    Cast_3 = Column(String)
    Director_1 = Column(String)
    IMDB_Rating = Column(Float(precision=2))
    Metascore = Column(Float(precision=2))
    created_at = Column(DateTime, default=func.now())
