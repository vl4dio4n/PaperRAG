from typing import List
from preprocess.chunkers.base_chunker import BaseChunker
from commons.secret_manager import SecretManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.schema import BaseNode


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        secret_manager: SecretManager,
        embed_model_type: str = "huggingface",
        model_name: str = "BAAI/bge-m3",
        breakpoint_percentile: int = 80,
        device: str = "cpu",
    ):
        self.percentile = breakpoint_percentile
        self.model_name = model_name

        if embed_model_type == "huggingface":
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name, trust_remote_code=True, device=device
            )
        elif embed_model_type == "gemini":
            self.embed_model = GeminiEmbedding(
                model_name=model_name, api_key=secret_manager.get_google_key()
            )
        else:
            raise ValueError(f"Unknown embedding type: {embed_model_type}")

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=breakpoint_percentile,
            embed_model=self.embed_model,
        )

    def get_strategy_name(self) -> str:
        clean_model = self.model_name.replace("/", "_").replace("-", "_")
        return f"semantic_{self.percentile}_{clean_model}"

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        print(f"Chunking with SemanticSplitter ({self.get_strategy_name()})...")
        return self._chunk_with_cache(documents, self.splitter)
