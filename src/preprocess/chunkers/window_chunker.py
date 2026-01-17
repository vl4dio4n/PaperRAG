from typing import List
from preprocess.chunkers.base_chunker import BaseChunker
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document
from llama_index.core.schema import BaseNode


class WindowChunker(BaseChunker):
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.splitter = SentenceWindowNodeParser(
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

    def get_strategy_name(self) -> str:
        return f"window_{self.window_size}"

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        print(f"Chunking with WindowSplitter (size={self.window_size})...")
        return self._chunk_with_cache(documents, self.splitter)
