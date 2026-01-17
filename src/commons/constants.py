import os
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


PDF_DIR = "../ad-papers-pdf"
MD_DIR = "../ad-papers-md"
CHUNKS_ROOT_DIR = "../ad-papers-chunked"

os.makedirs(MD_DIR, exist_ok=True)
os.makedirs(CHUNKS_ROOT_DIR, exist_ok=True)


BGE_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
BGE_EMBEDDING_MODEL_CLEAN_NAME = BGE_EMBEDDING_MODEL_NAME.replace("/", "_").replace(
    "-", "_"
)
GEMINI_EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
GEMINI_EMBEDDING_MODEL_CLEAN_NAME = GEMINI_EMBEDDING_MODEL_NAME.replace(
    "/", "_"
).replace("-", "_")

BGE_CHUNKING_SRATEGY_NAME = f"semantic_80_{BGE_EMBEDDING_MODEL_CLEAN_NAME}"
GEMINI_CHUNKING_SRATEGY_NAME = f"semantic_80_{GEMINI_EMBEDDING_MODEL_CLEAN_NAME}"
WINDOW_CHUNKING_SRATEGY_NAME = "window_5"

FILES = [
    os.path.join(PDF_DIR, "extended_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "extended_kmeans_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "functional_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "generalized_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "kernel_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "kmeans_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "probabilistic_generalization_of_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "randomised_choices_in_isolation_forest.pdf"),
    os.path.join(PDF_DIR, "scoring_isolation_forest.pdf"),
]
META = [
    {"title": "Extended Isolation Forest", "authors": "Hariri et al.", "year": 2021},
    {
        "title": "Extended K-Means Isolation Forest",
        "authors": "Vlad Birsan",
        "year": 2025,
    },
    {"title": "Functional Isolation Forest", "authors": "Staerman", "year": 2019},
    {
        "title": "Generalized isolation forest for anomaly detection",
        "authors": "Lesouple et al.",
        "year": 2021,
    },
    {
        "title": "Hyperspectral anomaly detection with kernel isolation forest",
        "authors": "Li et al.",
        "year": 2019,
    },
    {
        "title": "K-means-based isolation forest",
        "authors": "Karczmarek et al.",
        "year": 2020,
    },
    {
        "title": "A probabilistic generalization of isolation forest",
        "authors": "Tokovarov,",
        "year": 2022,
    },
    {
        "title": "Revisiting randomized choices in isolation forests",
        "authors": "Cortes et al.",
        "year": 2021,
    },
    {
        "title": "Distribution and volume based scoring for Isolation Forests",
        "authors": "Dhouib et al.",
        "year": 2023,
    },
]

QUESTIONS = [
    # --- ANSWERABLE (14 Questions) ---
    # 1. From 'Extended Isolation Forest' (Hariri et al.)
    {
        "question": "What specific artifact does the standard Isolation Forest produce in anomaly score heat maps that Extended Isolation Forest aims to fix?",
        "label": "ANSWERABLE",
    },
    # 2. From 'Extended Isolation Forest' (Hariri et al.) - (Your original question)
    {
        "question": "How does Extended Isolation Forest fix the bias issues found in the standard algorithm?",
        "label": "ANSWERABLE",
    },
    # 3. From 'Functional Isolation Forest' (Staerman et al.)
    {
        "question": "How does Functional Isolation Forest (FIF) project data using a dictionary and scalar products?",
        "label": "ANSWERABLE",
    },
    # 4. From 'Hyperspectral Anomaly Detection with Kernel Isolation Forest' (Li et al.)
    {
        "question": "Why are anomalies assumed to be more susceptible to isolation in the kernel space according to the Kernel Isolation Forest paper?",
        "label": "ANSWERABLE",
    },
    # 5. From 'Generalized Isolation Forest' (Lesouple et al.)
    {
        "question": "How does Generalized Isolation Forest (GIF) improve upon Extended Isolation Forest regarding empty branches?",
        "label": "ANSWERABLE",
    },
    # 6. From 'K-Means-based Isolation Forest' (Karczmarek et al.)
    {
        "question": "How does the K-Means Isolation Forest algorithm combine the partition strategy with the K-Means clustering algorithm?",
        "label": "ANSWERABLE",
    },
    # 7. From 'Extended K-Means Isolation Forest' (Birsan, 2025)
    {
        "question": "What are the two hybrid algorithms introduced in the Extended K-Means Isolation Forest paper?",
        "label": "ANSWERABLE",
    },
    # 8. From 'Probabilistic Generalization of Isolation Forest' (Tokovarov et al.)
    {
        "question": "How does the Probabilistic Generalization of Isolation Forest (PGIF) use segment-cumulated probability?",
        "label": "ANSWERABLE",
    },
    # 9. From 'Distribution and volume based scoring' (Dhouib et al.)
    {
        "question": "How does the Rényi divergence relate to the aggregation functions in distribution-based scoring for Isolation Forests?",
        "label": "ANSWERABLE",
    },
    # 10. From 'Revisiting randomized choices' (Cortes et al.)
    {
        "question": "According to the 'Revisiting randomized choices' paper, how does non-uniform random splitting affect the detection of clustered outliers?",
        "label": "ANSWERABLE",
    },
    # 11. From 'Kernel Isolation Forest'
    {
        "question": "What is the specific application domain (type of images) that the Kernel Isolation Forest is designed to analyze?",
        "label": "ANSWERABLE",
    },
    # 12. From 'Extended K-Means Isolation Forest'
    {
        "question": "Which benchmark metrics were used to evaluate the Extended K-Means Isolation Forest on the 13 datasets?",
        "label": "ANSWERABLE",
    },
    # 13. From 'Functional Isolation Forest'
    {
        "question": "What is the 'visual elbow rule' used for in the context of Functional Isolation Forest experiments?",
        "label": "ANSWERABLE",
    },
    # 14. From 'Generalized Isolation Forest'
    {
        "question": "What is the main advantage of Generalized Isolation Forest (GIF) over Extended Isolation Forest (EIF) in terms of computation time?",
        "label": "ANSWERABLE",
    },
    # --- NO DATA (4 Questions) ---
    {
        "question": "How does the performance of Isolation Forest compare to an LSTM-based Autoencoder on time-series data?",
        "label": "NO_DATA",
    },
    {
        "question": "What are the specific latency requirements for deploying Isolation Forest on an Arduino or edge device?",
        "label": "NO_DATA",
    },
    {
        "question": "How can I implement the Isolation Forest algorithm using the H2O.ai library in R?",
        "label": "NO_DATA",
    },
    {
        "question": "Does the 'Deep Isolation Forest' variant use Convolutional Neural Networks for feature extraction?",
        "label": "NO_DATA",
    },
    # --- UNRELATED (2 Questions) ---
    {"question": "What is the best recipe for pizza?", "label": "UNRELATED"},
    {"question": "Who won the FIFA World Cup in 2022?", "label": "UNRELATED"},
]
