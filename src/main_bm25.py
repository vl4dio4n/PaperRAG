import fitz
import nltk
import string

from commons.constants import FILES, META, QUESTIONS
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class ClassicalParser:
    def parse(
        self, file_path: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        print(f"Parsing {file_path} with Fitz...")
        doc = fitz.open(file_path)
        chunks = []

        base_metadata = metadata if metadata else {}

        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

            for p in paragraphs:
                chunk_meta = base_metadata.copy()
                chunk_meta["source"] = file_path
                chunk_meta["page"] = page_num + 1

                chunks.append({"text": p, "metadata": chunk_meta})

        print(f"Found {len(chunks)} paragraphs.")
        return chunks


class ClassicalRetriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus = [chunk["text"] for chunk in chunks]

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        print("Building BM25 index (stopwords + lemmatization)...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)

        clean_tokens = []
        for w in tokens:
            if w not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(w, pos="v")
                clean_tokens.append(lemma)

        return clean_tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = self._tokenize(query)

        print(f"Query tokens: {tokenized_query}")

        scores = self.bm25.get_scores(tokenized_query)

        top_n_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in top_n_indices:
            if scores[i] > 0:
                result_item = self.chunks[i].copy()
                result_item["score"] = scores[i]
                results.append(result_item)
        return results


class ClassicalPipeline:
    def __init__(self, file_paths: List[str], metadatas: List[Dict[str, Any]]):
        self.parser = ClassicalParser()
        self.all_chunks = []

        for path, meta in zip(file_paths, metadatas):
            file_chunks = self.parser.parse(path, metadata=meta)
            self.all_chunks.extend(file_chunks)

        self.retriever = ClassicalRetriever(self.all_chunks)

    def run(self, query: str, top_k: int = 3):
        print(f"\nClassical search for: '{query}'")
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant documents found (0 keyword matches)."

        output = ""
        for i, res in enumerate(results):
            meta = res["metadata"]

            title = meta.get("title", "Unknown Title")
            author = meta.get("authors", "Unknown Author")
            year = meta.get("year", "n.d.")

            output += f"--- Result {i + 1} (BM25 Score: {res['score']:.2f}) ---\n"
            output += f"Source: [{title}, {author}, {year}]\n"
            output += f"File: {meta['source']} (Page {meta['page']})\n"
            output += f"Content: {res['text'][:300]}...\n\n"

        return output


if __name__ == "__main__":
    classical_pipeline = ClassicalPipeline(FILES, META)
    for q in QUESTIONS:
        print(classical_pipeline.run(q["question"]))
