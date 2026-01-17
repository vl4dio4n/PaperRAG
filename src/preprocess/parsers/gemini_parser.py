import os

from typing import List
from google import genai
from preprocess.parsers.base_parser import BaseParser
from commons.secret_manager import SecretManager
from commons.constants import MD_DIR
from llama_index.core import Document


class GeminiParser(BaseParser):
    def __init__(
        self,
        secret_manager: SecretManager,
        model_name: str = "models/gemini-pro-latest",
    ):
        self.client = genai.Client(api_key=secret_manager.get_google_key())
        self.model_name = model_name
        self.prompt = """
        The provided document is a scientific research paper.
        Your goal is to extract ALL text, tables, and formulas into Markdown format.
        1. Transcribe text STRICTLY VERBATIM. Do not summarize, shorten, or rephrase.
        2. Do not skip any sections, subsections, or paragraphs, even if they look dense.
        3. Maintain the reading order of the paragraphs and columns.
        4. Don't include the figures/images. Instead, provide a description of the content of the figure, along with the caption.
        5. Exclude the headers and footers of the pages.
        """

    def parse(self, file_path: str, metadata: dict = None) -> List[Document]:
        base_name = os.path.basename(file_path).replace(".pdf", ".md")
        cache_path = os.path.join(MD_DIR, base_name)

        text_content = ""

        if os.path.exists(cache_path):
            print(f"Loading cached markdown for: {base_name}")
            with open(cache_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        else:
            print(f"Parsing with Gemini (API Call): {file_path}...")
            file_ref = self.client.files.upload(file=file_path)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=[file_ref, self.prompt]
                )
                text_content = response.text

                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
            finally:
                self.client.files.delete(name=file_ref.name)

        doc_metadata = metadata or {}
        doc_metadata["file_path"] = file_path

        return [Document(text=text_content, metadata=doc_metadata)]
