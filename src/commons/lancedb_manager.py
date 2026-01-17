import lancedb

from commons.secret_manager import SecretManager
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


class LanceDBManager:
    def __init__(
        self,
        secret_manager: SecretManager,
        db_uri: str = "../lancedb_data",
        embed_model_type: str = "huggingface",
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self.db_uri = db_uri
        self.db = lancedb.connect(db_uri)

        if embed_model_type == "huggingface":
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name, trust_remote_code=True, device=device
            )
        elif embed_model_type == "gemini":
            self.embed_model = GeminiEmbedding(
                model_name=model_name, api_key=secret_manager.get_google_key()
            )

        self.model_name_clean = model_name.replace("/", "_").replace("-", "_")

    def store_data(
        self, nodes: List[BaseNode], chunking_strategy_name: str
    ) -> VectorStoreIndex:
        table_name = f"{chunking_strategy_name}_embed_{self.model_name_clean}"
        print(f"--- Accessing Table: {table_name} ---")

        existing_tables = self.db.list_tables().tables

        if table_name in existing_tables:
            print("Table exists. Loading index...")
            vector_store = LanceDBVectorStore(uri=self.db_uri, table_name=table_name)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
            )
        else:
            print("Table not found. Creating and Indexing (this takes time)...")
            vector_store = LanceDBVectorStore(
                uri=self.db_uri, table_name=table_name, mode="overwrite"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(
                nodes, storage_context=storage_context, embed_model=self.embed_model
            )
            print("Indexing complete.")

        return index

    def get_retriever(self, index: VectorStoreIndex, similarity_top_k: int = 5):
        return index.as_retriever(similarity_top_k=similarity_top_k)
