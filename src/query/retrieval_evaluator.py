import json

from typing import List, Tuple
from google import genai
from llama_index.core.schema import BaseNode
from commons.secret_manager import SecretManager


class RetrievalEvaluator:
    """
    Analyzes the retrieved chunks to determine if we can answer.
    States:
    1. ANSWERABLE: Good chunks found.
    2. NO_DATA: Query is relevant to domain (Isolation Forest), but specific details are missing.
    3. UNRELATED: Query is about "cooking" or "weather".
    """

    def __init__(
        self,
        secret_manager: SecretManager,
        model_name: str = "models/gemini-pro-latest",
    ):
        self.client = genai.Client(api_key=secret_manager.get_google_key())
        self.model_name = model_name

    def evaluate(self, query: str, nodes: List[BaseNode]) -> Tuple[str, str]:
        context_parts = []
        for n in nodes:
            content_text = n.metadata.get("window", n.text)
            context_parts.append(content_text)
        context_text = "\n\n".join(context_parts)

        prompt = f"""
        You are the "Gatekeeper" for a Research Assistant about Anomaly Detection (Isolation Forests).
        Your job is to classify the relationship between the USER QUERY and the RETRIEVED CONTEXT.

        USER QUERY: {query}

        RETRIEVED CONTEXT:
        {context_text}

        Task: Analyze the inputs and output ONE of the following JSON strings:

        1. If the query is completely unrelated to Computer Science/Anomaly Detection (e.g. "How to cook pasta", "What is the weather"):
           {{"status": "UNRELATED", "reason": "The user is asking about [Topic] which is outside the scope of this research assistant."}}

        2. If the query IS related to the domain, but the Retrieved Context DOES NOT contain the answer:
           {{"status": "NO_DATA", "reason": "The query is relevant, but the provided papers do not discuss this specific detail."}}

        3. If the Retrieved Context contains the answer:
           {{"status": "ANSWERABLE", "reason": "Context contains sufficient information."}}

        OUTPUT JSON ONLY:
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        result = json.loads(response.text)
        print(f"Evaluation Status: {result['status']}")
        return result["status"], result["reason"]
