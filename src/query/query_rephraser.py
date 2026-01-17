from commons.secret_manager import SecretManager
from google import genai


class QueryRephraser:
    def __init__(
        self,
        secret_manager: SecretManager,
        model_name: str = "models/gemini-pro-latest",
    ):
        self.client = genai.Client(api_key=secret_manager.get_google_key())
        self.model_name = model_name

    def rephrase(self, query: str) -> str:
        prompt = f"""
        You are an AI research assistant. The user is asking a question about "Isolation Forests" or anomaly detection.
        Rephrase the following question to be more specific and optimized for a vector search engine.
        - Keep the core intent.
        - Expand technical acronyms (e.g., "IF" -> "Isolation Forest").
        - If the query is a simple keyword, turn it into a full sentence.

        Original Query: {query}
        Rephrased Query:
        """
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        new_query = response.text.strip()
        print(f"Rephrased: '{query}' -> '{new_query}'")
        return new_query
