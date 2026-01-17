from typing import List
from google import genai
from llama_index.core.schema import BaseNode
from commons.secret_manager import SecretManager


class ResponseGenerator:
    def __init__(
        self,
        secret_manager: SecretManager,
        model_name: str = "models/gemini-pro-latest",
    ):
        self.client = genai.Client(api_key=secret_manager.get_google_key())
        self.model_name = model_name

    def generate(self, query: str, nodes: List[BaseNode]) -> str:
        context_str = ""
        for i, node in enumerate(nodes):
            meta = node.metadata

            title = meta.get("title", "Unknown Title")
            author = meta.get("authors", "Unknown Authors")
            year = meta.get("year", "n.d.")
            citation_tag = f"[{title}, {author}, {year}]"

            content_text = meta.get("window", node.text)

            context_str += f"--- Source {i + 1} {citation_tag} ---\n{content_text}\n\n"

        prompt = f"""
        You are a Research Assistant. Answer the question using ONLY the provided context.

        Rules:
        1. Cite your sources using the format [Title, Author, Year] provided in the header of each source.
        2. Do not hallucinate information not present in the text.
        3. If the context has multiple papers, synthesize them.

        Context:
        {context_str}

        Question: {query}

        Answer:
        """

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return response.text
