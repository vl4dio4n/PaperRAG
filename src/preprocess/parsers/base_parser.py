from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str, metadata: dict = None) -> List[Document]:
        pass
