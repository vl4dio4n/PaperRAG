from commons.constants import (
    DEVICE,
    FILES,
    META,
    BGE_EMBEDDING_MODEL_NAME,
    GEMINI_EMBEDDING_MODEL_NAME,
)
from commons.lancedb_manager import LanceDBManager
from commons.secret_manager import SecretManager
from preprocess.parsers.gemini_parser import GeminiParser
from preprocess.chunkers.semantic_chunker import SemanticChunker
from preprocess.chunkers.window_chunker import WindowChunker
from preprocess.pipelines.preprocessing_pipeline import PreprocessingPipeline


if __name__ == "__main__":
    secrets = SecretManager()
    parser = GeminiParser(secrets)
    bge_chunker = SemanticChunker(
        secrets,
        embed_model_type="huggingface",
        model_name=BGE_EMBEDDING_MODEL_NAME,
        breakpoint_percentile=80,
        device=DEVICE,
    )
    gemini_chunker = SemanticChunker(
        secrets,
        embed_model_type="gemini",
        model_name=GEMINI_EMBEDDING_MODEL_NAME,
        breakpoint_percentile=80,
        device=DEVICE,
    )
    window_chunker = WindowChunker(window_size=5)
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

    # Preprocessing corpus using semantic chunking with bge-m3 model and bge-m3 embeddings
    pipeline = PreprocessingPipeline(parser, bge_chunker, bge_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)

    # Preprocessing corpus using semantic chunking with bge-m3 model and gemini embeddings
    pipeline = PreprocessingPipeline(parser, bge_chunker, gemini_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)

    # Preprocessing corpus using semantic chunking with gemini model and bge-m3 embeddings
    pipeline = PreprocessingPipeline(parser, gemini_chunker, bge_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)

    # Preprocessing corpus using semantic chunking with gemini model and gemini embeddings
    pipeline = PreprocessingPipeline(parser, gemini_chunker, gemini_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)

    # Preprocessing corpus using window chunking and bge-m3 embeddings
    pipeline = PreprocessingPipeline(parser, window_chunker, bge_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)

    # Preprocessing corpus using window chunking and gemini embeddings
    pipeline = PreprocessingPipeline(parser, window_chunker, gemini_db_manager)
    print("Starting preprocessing pipeline...")
    _ = pipeline.run(FILES, META)
