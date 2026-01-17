from typing import List
from commons.lancedb_manager import LanceDBManager
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode


class RetrieverModule:
    def __init__(
        self, db_manager: LanceDBManager, chunking_strategy: str, embed_model_name: str
    ):
        self.db_manager = db_manager

        clean_model = embed_model_name.replace("/", "_").replace("-", "_")
        self.table_name = f"{chunking_strategy}_embed_{clean_model}"

        print(f"Connecting Retriever to table: {self.table_name}")

        vector_store = LanceDBVectorStore(
            uri=db_manager.db_uri, table_name=self.table_name
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=db_manager.embed_model
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[BaseNode]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} raw chunks.")
        return nodes
