from query.query_rephraser import QueryRephraser
from query.retriever_module import RetrieverModule
from query.gemini_reranker import GeminiReranker
from query.retrieval_evaluator import RetrievalEvaluator
from query.response_generator import ResponseGenerator
from typing import Optional


class QueryPipeline:
    def __init__(
        self,
        rephraser: Optional[QueryRephraser],
        retriever: RetrieverModule,
        reranker: Optional[GeminiReranker],
        evaluator: RetrievalEvaluator,
        generator: ResponseGenerator,
    ):
        self.rephraser = rephraser
        self.retriever = retriever
        self.reranker = reranker
        self.evaluator = evaluator
        self.generator = generator

    def run(
        self, user_query: str, use_rephrasing: bool = True, use_reranking: bool = True
    ):
        print(f"\n--- Starting pipeline for: '{user_query}' ---")

        search_query = user_query
        if self.rephraser and use_rephrasing:
            search_query = self.rephraser.rephrase(user_query)

        retrieved_nodes = self.retriever.retrieve(
            search_query, top_k=20 if self.reranker else 15
        )

        final_nodes = retrieved_nodes
        if self.reranker and use_reranking:
            final_nodes = self.reranker.rerank(search_query, retrieved_nodes, top_n=10)

        status, reason = self.evaluator.evaluate(user_query, final_nodes)

        if status == "UNRELATED":
            return f"**Query Rejected:** {reason}\n(I only answer questions about the provided research papers.)"

        elif status == "NO_DATA":
            return f"**No Information Found:** {reason}\n(I searched the database but couldn't find specific details on this.)"

        print("Generating answer...")
        answer = self.generator.generate(user_query, final_nodes)
        return answer
