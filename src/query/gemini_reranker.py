from typing import List
from google import genai
from llama_index.core.schema import BaseNode
from commons.secret_manager import SecretManager


class GeminiReranker:
    def __init__(
        self,
        secret_manager: SecretManager,
        model_name: str = "models/gemini-pro-latest",
    ):
        self.client = genai.Client(api_key=secret_manager.get_google_key())
        self.model_name = model_name

    def rerank(
        self, query: str, nodes: List[BaseNode], top_n: int = 3
    ) -> List[BaseNode]:
        if not nodes:
            return []

        candidates_text = ""
        for i, node in enumerate(nodes):
            content_text = node.metadata.get("window", node.text)
            candidates_text += f"ID: {i}\nContent: {content_text}...\n\n"

        prompt = f"""
        You are a relevance ranking system.
        Query: "{query}"

        Below are candidate text chunks retrieved for this query.
        Rank them by relevance to the query.
        Return ONLY the IDs of the top {top_n} most relevant chunks, separated by commas.
        If a chunk is completely irrelevant, exclude it.

        Candidates:
        {candidates_text}

        Result IDs:
        """

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        try:
            indices_str = response.text.strip().replace("Result IDs:", "")
            selected_indices = [
                int(idx.strip())
                for idx in indices_str.split(",")
                if idx.strip().isdigit()
            ]

            reranked_nodes = [nodes[i] for i in selected_indices if i < len(nodes)]
            print(f"Reranked: Kept {len(reranked_nodes)}/{len(nodes)} chunks.")
            return reranked_nodes

        except Exception as e:
            print(f"Reranking failed ({e}), returning original top {top_n}.")
            return nodes[:top_n]
