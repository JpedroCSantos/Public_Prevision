from abc import ABC, abstractmethod
from typing import Dict


class MovieAPI(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def search_movie(self, row: Dict) -> Dict:
        raise NotImplementedError("Method 'search_movie' not implemented")

    @abstractmethod
    def build_object():
        raise NotImplementedError("Method 'build_object' not implemented")

    @abstractmethod
    def get_api_dict():
        raise NotImplementedError("Method 'get_api_dict' not implemented")
