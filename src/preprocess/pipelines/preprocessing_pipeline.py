from typing import List
from preprocess.chunkers.base_chunker import BaseChunker
from preprocess.parsers.base_parser import BaseParser
from commons.lancedb_manager import LanceDBManager


class PreprocessingPipeline:
    def __init__(
        self, parser: BaseParser, chunker: BaseChunker, db_manager: LanceDBManager
    ):
        self.parser = parser
        self.chunker = chunker
        self.db_manager = db_manager

    def run(self, file_paths: List[str], citation_metadata: List[dict]):
        all_documents = []

        for path, meta in zip(file_paths, citation_metadata):
            print(f"Processing: {path}...")
            docs = self.parser.parse(path, metadata=meta)
            all_documents.extend(docs)

        nodes = self.chunker.chunk(all_documents)

        strategy_name = self.chunker.get_strategy_name()
        index = self.db_manager.store_data(nodes, strategy_name)

        return index
