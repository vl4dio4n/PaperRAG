import os
from dotenv import load_dotenv

load_dotenv()


class SecretManager:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
        if self.llama_cloud_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = self.llama_cloud_key

    def get_google_key(self):
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file or system environment variables."
            )
        return self.google_api_key

    def get_llama_key(self):
        if not self.llama_cloud_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not found. Please set it in your .env file or system environment variables."
            )
        return self.llama_cloud_key
