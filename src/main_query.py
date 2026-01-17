from commons.lancedb_manager import LanceDBManager
from commons.secret_manager import SecretManager
from commons.constants import (
    DEVICE,
    QUESTIONS,
    BGE_EMBEDDING_MODEL_NAME,
    BGE_CHUNKING_SRATEGY_NAME,
    GEMINI_EMBEDDING_MODEL_NAME,
    GEMINI_CHUNKING_SRATEGY_NAME,
    WINDOW_CHUNKING_SRATEGY_NAME,
)
from query.query_rephraser import QueryRephraser
from query.retriever_module import RetrieverModule
from query.retrieval_evaluator import RetrievalEvaluator
from query.gemini_reranker import GeminiReranker
from query.response_generator import ResponseGenerator
from query.query_pipeline import QueryPipeline
from typing import List, Dict


def answer_questions(
    questions: List[Dict[str, str]],
    pipeline: QueryPipeline,
    use_rephrasing: bool,
    use_reranking: bool,
) -> None:
    for q_data in questions:
        user_query = q_data["question"]
        expected_label = q_data["label"]

        print(f"\nProcessing: '{user_query}'")

        response = pipeline.run(
            user_query=user_query,
            use_rephrasing=use_rephrasing,
            use_reranking=use_reranking,
        )

        print(f"Response:\n{response}")
        print(f"Expected Label: {expected_label}")
        print("-" * 60)


if __name__ == "__main__":
    secrets = SecretManager()
    rephraser = QueryRephraser(secrets)

    bge_db_manager = LanceDBManager(
        secrets,
        embed_model_type="huggingface",
        model_name=BGE_EMBEDDING_MODEL_NAME,
        device=DEVICE,
    )
    gemini_db_manager = LanceDBManager(
        secrets,
        embed_model_type="gemini",
        model_name=GEMINI_EMBEDDING_MODEL_NAME,
        device=DEVICE,
    )

    bge_chunker_bge_embed_retriever_mod = RetrieverModule(
        db_manager=bge_db_manager,
        chunking_strategy=BGE_CHUNKING_SRATEGY_NAME,
        embed_model_name=BGE_EMBEDDING_MODEL_NAME,
    )
    bge_chunker_gemini_embed_retriever_mod = RetrieverModule(
        db_manager=gemini_db_manager,
        chunking_strategy=BGE_CHUNKING_SRATEGY_NAME,
        embed_model_name=GEMINI_EMBEDDING_MODEL_NAME,
    )
    gemini_chunker_bge_embed_retriever_mod = RetrieverModule(
        db_manager=bge_db_manager,
        chunking_strategy=GEMINI_CHUNKING_SRATEGY_NAME,
        embed_model_name=BGE_EMBEDDING_MODEL_NAME,
    )
    gemini_chunker_gemini_embed_retriever_mod = RetrieverModule(
        db_manager=gemini_db_manager,
        chunking_strategy=GEMINI_CHUNKING_SRATEGY_NAME,
        embed_model_name=GEMINI_EMBEDDING_MODEL_NAME,
    )
    window_chunker_bge_embed_retriever_mod = RetrieverModule(
        db_manager=bge_db_manager,
        chunking_strategy=WINDOW_CHUNKING_SRATEGY_NAME,
        embed_model_name=BGE_EMBEDDING_MODEL_NAME,
    )
    window_chunker_gemini_embed_retriever_mod = RetrieverModule(
        db_manager=gemini_db_manager,
        chunking_strategy=WINDOW_CHUNKING_SRATEGY_NAME,
        embed_model_name=GEMINI_EMBEDDING_MODEL_NAME,
    )

    reranker = GeminiReranker(secrets)
    evaluator = RetrievalEvaluator(secrets)
    generator = ResponseGenerator(secrets)

    bge_chunker_bge_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=bge_chunker_bge_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )
    bge_chunker_gemini_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=bge_chunker_gemini_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )
    gemini_chunker_bge_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=gemini_chunker_bge_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )
    gemini_chunker_gemini_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=gemini_chunker_gemini_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )
    window_chunker_bge_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=window_chunker_bge_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )
    window_chunker_gemini_embed_query_pipeline = QueryPipeline(
        rephraser=rephraser,
        retriever=window_chunker_gemini_embed_retriever_mod,
        reranker=reranker,
        evaluator=evaluator,
        generator=generator,
    )

    #################################################################################
    ### Question answering: semantic chunker with bge-m3 model, bge-m3 embeddings ###
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks rearanking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )

    #################################################################################
    ### Question answering: semantic chunker with bge-m3 model, gemini embeddings ###
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=bge_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )

    #################################################################################
    ### Question answering: semantic chunker with gemini model, bge-m3 embeddings ###
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )

    #################################################################################
    ### Question answering: semantic chunker with gemini model, gemini embeddings ###
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=gemini_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )

    #################################################################################
    ############# Question answering: window chunker, bge-m3 embeddings #############
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_bge_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_bge_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )

    #################################################################################
    ############# Question answering: window chunker, gemini embeddings #############
    #################################################################################

    # Question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=True,
    )

    # Question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_gemini_embed_query_pipeline,
        use_rephrasing=True,
        use_reranking=False,
    )

    # No question rephrasing, chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=True,
    )

    # No question rephrasing, no chunks reranking
    answer_questions(
        questions=QUESTIONS,
        pipeline=window_chunker_gemini_embed_query_pipeline,
        use_rephrasing=False,
        use_reranking=False,
    )
