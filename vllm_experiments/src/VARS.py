# --- Constants ---
GRAPH_FILE = "large_knowledge_graph.graphml"  # "large_knowledge_graph_modified.graphml"
# GRAPH_FILE = "large_knowledge_graph.graphml"
GRAPH_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
SIMILARITY_THRESHOLD = 0.8
MAX_TEMPLATE_LENGTH = 8
FIRST_STEP_WEIGHT_QUERY_SIM = 0.75
FIRST_STEP_WEIGHT_S0_BONUS = 0.25
MAX_NEW_TOKENS = 18000
MAX_NON_THINKING_TOKENS = 0
THINKING_INTERVEN = False
SIMILARITY_THRESHOLD_RERANK = 0.7
