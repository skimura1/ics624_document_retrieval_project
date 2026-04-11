from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def fit(self, passages_text: list[str]):
        pass

    @abstractmethod
    def query(self, query: str):
        pass
