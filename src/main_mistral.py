import torch

from commons.constants import DEVICE, QUESTIONS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class BaselineMistral:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cpu",
    ):
        print(f"Loading baseline Model: {model_name}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        print("Baseline model loaded.")

    def generate_answer(self, query: str) -> str:
        system_prompt = """
        You are a strict Research Assistant specializing ONLY in Anomaly Detection and Isolation Forests.

        YOUR RULES:
        1. You answer questions strictly based on technical knowledge of Isolation Forests.
        2. NEGATIVE CONSTRAINT: If the user asks about unrelated topics (like cooking, weather, sports, or general life advice), you must REFUSE.
           - YOU MUST NOT provide the requested information "anyway."
           - YOU MUST NOT say "However, here is..."
           - You simply state: "I cannot answer this question as it is unrelated to Anomaly Detection."
        3. If the user asks a technical question you don't know, say "I do not have sufficient data."
        """

        full_prompt = f"{system_prompt}\n\nUser Question: {query}"

        messages = [{"role": "user", "content": full_prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()


if __name__ == "__main__":
    baseline_mistral = BaselineMistral(device=DEVICE)

    for q_data in QUESTIONS:
        query = q_data["question"]
        expected_label = q_data["label"]

        print(f"Question: {query}")
        print(f"Expected Category: {expected_label}")
        print("-" * 60)
        print("[BASELINE MISTRAL RESPONSE]")
        baseline_response = baseline_mistral.generate_answer(query)
        print(baseline_response)
        print(f"\n{'=' * 60}\n")
