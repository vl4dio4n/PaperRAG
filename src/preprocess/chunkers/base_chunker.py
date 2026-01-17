import os
import json

from commons.constants import CHUNKS_ROOT_DIR
from typing import List
from abc import ABC, abstractmethod
from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

    def _chunk_with_cache(self, documents: List[Document], splitter) -> List[BaseNode]:
        strategy_name = self.get_strategy_name()
        strategy_dir = os.path.join(CHUNKS_ROOT_DIR, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)

        all_nodes = []

        for doc in documents:
            file_path = doc.metadata.get("file_path")

            if not file_path:
                all_nodes.extend(splitter.get_nodes_from_documents([doc]))
                continue

            base_name = os.path.basename(file_path).replace(".pdf", ".json")
            cache_path = os.path.join(strategy_dir, base_name)

            if os.path.exists(cache_path):
                print(f"Loading cached chunks for {base_name} ({strategy_name})...")
                with open(cache_path, "r", encoding="utf-8") as f:
                    nodes_data = json.load(f)
                    all_nodes.extend([TextNode.from_dict(n) for n in nodes_data])
            else:
                print(f"Computing chunks for {base_name} ({strategy_name})...")
                nodes = splitter.get_nodes_from_documents([doc])

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump([n.to_dict() for n in nodes], f)

                all_nodes.extend(nodes)

        return all_nodes
