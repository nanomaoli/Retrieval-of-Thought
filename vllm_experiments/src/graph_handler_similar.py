from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import logging
import time
import torch
import requests
from .VARS_similar import *


def load_knowledge_graph():
    try:
        G = nx.read_graphml(GRAPH_FILE)
        logging.info(f"[similar] Knowledge graph loaded successfully from {GRAPH_FILE}")
        return G
    except FileNotFoundError:
        logging.error(
            f"Error: Graph file '{GRAPH_FILE}' not found. Please ensure the file exists."
        )
        exit()
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        exit()


def precompute_node_embeddings(G: nx.Graph, embedding_model: SentenceTransformer):
    node_texts = {}
    node_embeddings_cache = {}
    try:
        logging.info("Precomputing embeddings for all graph nodes...")
        all_node_ids = list(G.nodes())
        all_node_data = [G.nodes[node_id] for node_id in all_node_ids]
        all_texts = [data.get("text", "") for data in all_node_data]

        node_texts_to_encode = [
            (node_id, text) for node_id, text in zip(all_node_ids, all_texts) if text
        ]
        texts_for_encoding = [item[1] for item in node_texts_to_encode]

        computed_embeddings = np.array([])
        if texts_for_encoding:
            logging.info(f"Encoding {len(texts_for_encoding)} non-empty node texts...")
            computed_embeddings = embedding_model.encode(
                texts_for_encoding,
                show_progress_bar=False,
                batch_size=8,
            )

        embedding_dim = (
            embedding_model.get_sentence_embedding_dimension()
            if computed_embeddings.size > 0
            else None
        )
        node_id_to_encode_index = {
            item[0]: i for i, item in enumerate(node_texts_to_encode)
        }

        for node_id in all_node_ids:
            text = G.nodes[node_id].get("text", "")
            node_texts[node_id] = text
            if text and node_id in node_id_to_encode_index:
                node_embeddings_cache[node_id] = computed_embeddings[
                    node_id_to_encode_index[node_id]
                ]
            else:
                node_embeddings_cache[node_id] = None

        logging.info(
            f"Embeddings computed and cached for {len(node_texts_to_encode)} nodes."
        )
    except Exception as e:
        logging.error(f"Error computing embeddings: {e}", exc_info=True)
        exit()
    return node_texts, node_embeddings_cache


def generate_dynamic_template(
    query_text: str,
    query_knowledge_tags: list[str],
    # query_template_type: str,
    G: nx.Graph,
    embedding_model: SentenceTransformer,
    node_embeddings_cache,
    node_texts,
):
    """
    Generates a dynamic problem-solving template based on the query using s0 bonus for first step.
    """
    logging.info(f"Received query: '{query_text}'")
    logging.info(f"Query knowledge tags: {query_knowledge_tags}")
    # logging.info(f"Query template type: {query_template_type}")

    if not query_text:
        logging.warning("Query text is empty. Cannot generate template.")
        return []

    try:
        query_embedding = embedding_model.encode(sentences=[query_text])[0]
    except Exception as e:
        logging.error(f"Error encoding query: {e}", exc_info=True)
        return []

    # 1. First Step Retrieval (Revised Logic with s0 Bonus)
    logging.info("Filtering nodes and calculating scores for the first step...")

    # Pre-filter nodes based on metadata
    pre_filtered_nodes = []
    start_time = time.perf_counter()
    for node_id, data in G.nodes(data=True):
        # Metadata checks
        # node_template_type = data.get("template_type", "")
        node_knowledge_tags_str = data.get("knowledge_tag", "")
        # if node_template_type != query_template_type:
        #     continue
        tag_match = True
        if query_knowledge_tags:
            node_knowledge_tags = [
                tag.strip() for tag in node_knowledge_tags_str.split(",") if tag.strip()
            ]
            tag_match = any(
                query_tag in node_knowledge_tags for query_tag in query_knowledge_tags
            )
        if not tag_match:
            continue
        # Embedding check
        if (
            node_id not in node_embeddings_cache
            or node_embeddings_cache[node_id] is None
        ):
            continue
        pre_filtered_nodes.append(node_id)

    if not pre_filtered_nodes:
        logging.warning(
            "No candidate nodes found matching the query tags/type and having valid embeddings."
        )
        return [], time.perf_counter() - start_time
    logging.info(
        f"Found {len(pre_filtered_nodes)} initial candidate nodes based on metadata."
    )

    # Prepare data for batch similarity calculation
    candidate_step_embeddings = []
    candidate_node_ids_for_sim = []
    for node_id in pre_filtered_nodes:
        step_emb = node_embeddings_cache.get(node_id)
        if isinstance(
            step_emb, (np.ndarray, torch.Tensor)
        ):  # Ensure it's a valid embedding
            candidate_step_embeddings.append(step_emb)
            candidate_node_ids_for_sim.append(node_id)
        else:
            logging.warning(
                f"Node {node_id} has invalid embedding type ({type(step_emb)}), skipping similarity."
            )

    if not candidate_node_ids_for_sim:
        logging.warning("No candidates remaining for similarity calculation.")
        return [], time.perf_counter() - start_time

    # Batch calculate similarities
    try:
        step_similarities = cosine_similarity(
            [query_embedding], np.array(candidate_step_embeddings)
        )[0]
    except ValueError as e:
        logging.error(
            f"Error during cosine similarity calculation: {e}. Check embedding dimensions.",
            exc_info=True,
        )
        return [], 0

    # Calculate combined score with s0 bonus and find the best node
    best_combined_score = -np.inf
    start_node = None
    start_node_similarity = -1.0  # Store raw step similarity for logging

    for i, node_id in enumerate(candidate_node_ids_for_sim):
        # Add boundary check for safety
        step_sim = step_similarities[i] if i < len(step_similarities) else -1.0

        # Ensure sim is a valid float
        if not isinstance(step_sim, (float, np.float32, np.float64, torch.Tensor)):
            logging.warning(
                f"Invalid similarity type for node {node_id}: {type(step_sim)}. Skipping score calculation."
            )
            continue

        # Calculate s0 bonus
        is_s0_bonus = 1.0 if node_id.endswith("_s0") else 0.0
        combined_score = (
            FIRST_STEP_WEIGHT_QUERY_SIM * step_sim
            + FIRST_STEP_WEIGHT_S0_BONUS * is_s0_bonus
        )
        logging.debug(
            f"  Candidate: {node_id}, StepSim: {step_sim:.4f}, Is_s0: {is_s0_bonus:.1f}, Combined: {combined_score:.4f}"
        )

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            start_node = node_id
            start_node_similarity = (
                step_sim  # Log the step similarity of the chosen node
            )

    if start_node is None:
        logging.error("Could not determine a start node after scoring.")
        return [], 0

    logging.info(
        f"Selected start node: '{start_node}' with Combined Score {best_combined_score:.4f} (StepSim: {start_node_similarity:.4f})"
    )

    # 2. Graph Traversal (Hybrid Logic: 0.5 query sim / 0.5 flow score)
    # This part remains the same as the previous version
    dynamic_template_nodes = [start_node]
    current_node = start_node

    logging.info(
        "Starting graph traversal with hybrid scoring (0.5 QuerySim + 0.5 FlowScore)..."
    )
    while len(dynamic_template_nodes) < MAX_TEMPLATE_LENGTH:
        potential_next_nodes = list(G.successors(current_node))

        # Filter out visited nodes and nodes without embeddings
        valid_candidates = [
            node
            for node in potential_next_nodes
            if node not in dynamic_template_nodes
            and node_embeddings_cache.get(node) is not None
        ]

        if not valid_candidates:
            logging.info("Termination: No valid, unvisited successor nodes found.")
            break

        # Find the sequential child node, if it exists
        sequential_child_node = None
        for _, target, data in G.out_edges(current_node, data=True):
            if data.get("edge_type") == "sequential":
                sequential_child_node = target
                break  # Assuming only one sequential child

        # Score candidates using the hybrid approach (0.5 / 0.5 for traversal)
        best_traversal_combined_score = -np.inf
        best_next_node = None
        best_node_query_similarity = (
            -1.0
        )  # Store the query similarity of the best node for termination check

        candidate_data_for_scoring = []
        candidate_indices = {}
        for i, node_id in enumerate(valid_candidates):
            emb = node_embeddings_cache.get(node_id)
            if isinstance(emb, np.ndarray):
                candidate_indices[node_id] = len(candidate_data_for_scoring)
                candidate_data_for_scoring.append(emb)
            else:
                logging.warning(
                    f"Traversal: Skipping candidate {node_id} due to invalid embedding type: {type(emb)}"
                )

        if not candidate_data_for_scoring:
            break

        # Calculate all query similarities at once
        try:
            query_similarities = cosine_similarity(
                [query_embedding], np.array(candidate_data_for_scoring)
            )[0]
        except ValueError as e:
            logging.error(
                f"Error during traversal cosine similarity calculation for node {current_node}: {e}. Check embedding dimensions.",
                exc_info=True,
            )
            break

        for candidate_node, original_index in candidate_indices.items():
            # Add boundary check for safety
            query_sim = (
                query_similarities[original_index]
                if original_index < len(query_similarities)
                else -1.0
            )

            if not isinstance(query_sim, (float, np.float32, np.float64)):
                logging.warning(
                    f"Invalid query similarity type for traversal candidate {candidate_node}: {type(query_sim)}. Skipping."
                )
                continue

            flow_score = (
                1.0
                if sequential_child_node is not None
                and candidate_node == sequential_child_node
                else 0.0
            )
            combined_score = 0.5 * query_sim + 0.5 * flow_score

            if combined_score > best_traversal_combined_score:
                best_traversal_combined_score = combined_score
                best_next_node = candidate_node
                best_node_query_similarity = query_sim

        if best_next_node is None:
            logging.info(
                "Termination: Could not determine a best next node during traversal."
            )
            break

        if (
            best_node_query_similarity < SIMILARITY_THRESHOLD
            and best_traversal_combined_score < SIMILARITY_THRESHOLD
        ):
            logging.info(
                f"Termination: Best next node's query similarity ({best_node_query_similarity:.4f}) \
                and combined score ({best_traversal_combined_score:.4f}) \
                are below threshold ({SIMILARITY_THRESHOLD})."
            )
            break

        dynamic_template_nodes.append(best_next_node)
        current_node = best_next_node

    if len(dynamic_template_nodes) == MAX_TEMPLATE_LENGTH:
        logging.info(
            f"Termination: Reached maximum template length ({MAX_TEMPLATE_LENGTH})."
        )

    # 3. Assemble Template
    final_template = [
        node_texts.get(node_id, f"Error: Text not found for {node_id}")
        for node_id in dynamic_template_nodes
    ]
    logging.info(f"Generated template with {len(final_template)} steps.")

    return final_template, time.perf_counter() - start_time


def generate_dynamic_template_rerank(
    query_text: str,
    query_knowledge_tags: list[str],
    G: nx.Graph,
    node_texts,
    reranker_url: str,
    reranker_model_name: str,
    reranker_instruction: str,
    batch_size: int = 64,
):
    """
    Generates a dynamic problem-solving template based on the query using s0 bonus for first step.
    """
    RERANKER_PREFIX = r'<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    RERANKER_SUFFIX = r"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    logging.info(f"Received query: '{query_text}'")
    logging.info(f"Query knowledge tags: {query_knowledge_tags}")

    if not query_text:
        logging.warning("Query text is empty. Cannot generate template.")
        return []

    query_template = r"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = r"<Document>: {doc}{suffix}"

    # Use a session to reuse connections
    session = requests.Session()

    # 1. First Step Retrieval (Revised Logic with s0 Bonus)
    logging.info("Filtering nodes and calculating scores for the first step...")

    # Pre-filter nodes based on metadata
    pre_filtered_nodes = {}
    start_time = time.perf_counter()
    for node_id, data in G.nodes(data=True):
        # Metadata checks
        node_knowledge_tags_str = data.get("knowledge_tag", "")
        tag_match = True
        if query_knowledge_tags:
            node_knowledge_tags = [
                tag.strip() for tag in node_knowledge_tags_str.split(",") if tag.strip()
            ]
            tag_match = any(
                query_tag in node_knowledge_tags for query_tag in query_knowledge_tags
            )
        if not tag_match:
            continue

        pre_filtered_nodes[node_id] = G.nodes[node_id].get("text", "")

    if not pre_filtered_nodes:
        logging.warning("No candidate nodes found matching the query tags/type.")
        return [], time.perf_counter() - start_time
    logging.info(
        f"Found {len(pre_filtered_nodes)} initial candidate nodes based on metadata."
    )

    candidate_node_ids_for_sim = list(pre_filtered_nodes.keys())
    candidate_texts = [
        pre_filtered_nodes[node_id] for node_id in candidate_node_ids_for_sim
    ]

    if not candidate_node_ids_for_sim:
        logging.warning("No candidates remaining for similarity calculation.")
        return [], time.perf_counter() - start_time

    queries = [
        query_template.format(
            prefix=RERANKER_PREFIX,
            instruction=reranker_instruction,
            query=query_text,
        )
    ] * len(candidate_texts)
    documents = [
        document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
        for doc in candidate_texts
    ]

    step_similarities = {}

    for i in range(0, len(candidate_texts), batch_size):
        batch_texts = candidate_texts[i : i + batch_size]
        batch_node_ids = candidate_node_ids_for_sim[i : i + batch_size]

        queries = [
            query_template.format(
                prefix=RERANKER_PREFIX,
                instruction=reranker_instruction,
                query=query_text,
            )
        ] * len(batch_texts)
        documents = [
            document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
            for doc in batch_texts
        ]

        response = session.post(
            reranker_url,
            json={
                "model": reranker_model_name,
                "text_1": queries,
                "text_2": documents,
                "truncate_prompt_tokens": -1,
            },
        )
        response_data = response.json()
        score_list = response_data.get("data", [])

        for item in score_list:
            # The index in the response is relative to the batch, so we map it back
            original_node_id = batch_node_ids[item["index"]]
            step_similarities[original_node_id] = item["score"]

    # Calculate combined score with s0 bonus and find the best node
    best_combined_score = -np.inf
    start_node = None
    start_node_similarity = -1.0  # Store raw step similarity for logging

    for node_id in candidate_node_ids_for_sim:
        step_sim = step_similarities.get(node_id, -1.0)

        # Calculate s0 bonus
        is_s0_bonus = 1.0 if node_id.endswith("_s0") else 0.0
        combined_score = (
            FIRST_STEP_WEIGHT_QUERY_SIM * step_sim
            + FIRST_STEP_WEIGHT_S0_BONUS * is_s0_bonus
        )
        logging.debug(
            f"  Candidate: {node_id}, StepSim: {step_sim:.4f}, Is_s0: {is_s0_bonus:.1f}, Combined: {combined_score:.4f}"
        )

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            start_node = node_id
            start_node_similarity = (
                step_sim  # Log the step similarity of the chosen node
            )

    if start_node is None:
        logging.error("Could not determine a start node after scoring.")
        return [], 0

    logging.info(
        f"Selected start node: '{start_node}' with Combined Score {best_combined_score:.4f} (StepSim: {start_node_similarity:.4f})"
    )

    # 2. Graph Traversal (Hybrid Logic - 0.5 query sim / 0.5 flow score)
    dynamic_template_nodes = [start_node]
    current_node = start_node

    logging.info(
        "Starting graph traversal with hybrid scoring (0.5 QuerySim + 0.5 FlowScore)..."
    )
    while len(dynamic_template_nodes) < MAX_TEMPLATE_LENGTH:
        potential_next_nodes = list(G.successors(current_node))

        # Filter out visited nodes
        valid_candidates = [
            node for node in potential_next_nodes if node not in dynamic_template_nodes
        ]

        if not valid_candidates:
            logging.info("Termination: No valid, unvisited successor nodes found.")
            break

        # Find the sequential child node, if it exists
        sequential_child_node = None
        for _, target, data in G.out_edges(current_node, data=True):
            if data.get("edge_type") == "sequential":
                sequential_child_node = target
                break  # Assuming only one sequential child

        # Score candidates using the hybrid approach (0.5 / 0.5 for traversal)
        best_traversal_combined_score = -np.inf
        best_next_node = None
        best_node_query_similarity = (
            -1.0
        )  # Store the query similarity of the best node for termination check

        candidate_texts_for_scoring = [
            node_texts.get(node_id, "") for node_id in valid_candidates
        ]

        if not candidate_texts_for_scoring:
            break

        # Calculate all query similarities at once using reranker API
        try:
            queries = [
                query_template.format(
                    prefix=RERANKER_PREFIX,
                    instruction=reranker_instruction,
                    query=query_text,
                )
            ] * len(candidate_texts_for_scoring)
            documents = [
                document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
                for doc in candidate_texts_for_scoring
            ]

            response = session.post(
                reranker_url,
                json={
                    "model": reranker_model_name,
                    "text_1": queries,
                    "text_2": documents,
                    "truncate_prompt_tokens": -1,
                },
            ).json()
            score_list = response.get("data", [])
            query_similarities = {
                valid_candidates[item["index"]]: item["score"] for item in score_list
            }

        except Exception as e:
            logging.error(
                f"Error during traversal reranker API call for node {current_node}: {e}. Check API service.",
                exc_info=True,
            )
            break

        for candidate_node in valid_candidates:
            query_sim = query_similarities.get(candidate_node, -1.0)

            if not isinstance(query_sim, (float, np.float32, np.float64)):
                logging.warning(
                    f"Invalid query similarity type for traversal candidate {candidate_node}: {type(query_sim)}. Skipping."
                )
                continue

            flow_score = (
                1.0
                if sequential_child_node is not None
                and candidate_node == sequential_child_node
                else 0.0
            )
            combined_score = 0.5 * query_sim + 0.5 * flow_score

            if combined_score > best_traversal_combined_score:
                best_traversal_combined_score = combined_score
                best_next_node = candidate_node
                best_node_query_similarity = query_sim

        if best_next_node is None:
            logging.info(
                "Termination: Could not determine a best next node during traversal."
            )
            break

        if best_node_query_similarity < SIMILARITY_THRESHOLD_RERANK:
            logging.info(
                f"Termination: Best next node's query similarity ({best_node_query_similarity:.4f}) is below threshold ({SIMILARITY_THRESHOLD_RERANK})."
            )
            break

        dynamic_template_nodes.append(best_next_node)
        current_node = best_next_node

    if len(dynamic_template_nodes) == MAX_TEMPLATE_LENGTH:
        logging.info(
            f"Termination: Reached maximum template length ({MAX_TEMPLATE_LENGTH})."
        )

    # 3. Assemble Template
    final_template = [
        node_texts.get(node_id, f"Error: Text not found for {node_id}")
        for node_id in dynamic_template_nodes
    ]
    logging.info(f"Generated template with {len(final_template)} steps.")

    return final_template, time.perf_counter() - start_time


def generate_dynamic_template_rerank_beam_single(
    query_text: str,
    query_knowledge_tags: list[str],
    G: nx.Graph,
    node_texts,
    reranker_url: str,
    reranker_model_name: str,
    reranker_instruction: str,
    batch_size: int = 128,
    beam_width: int = 3,
):
    """
    Generates a dynamic problem-solving template based on the query using s0 bonus for first step.
    """
    RERANKER_PREFIX = r'<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    RERANKER_SUFFIX = r"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    logging.info(f"Received query: '{query_text}'")
    logging.info(f"Query knowledge tags: {query_knowledge_tags}")

    if not query_text:
        logging.warning("Query text is empty. Cannot generate template.")
        return [], 0

    query_template = r"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = r"<Document>: {doc}{suffix}"

    # Use a session to reuse connections
    session = requests.Session()

    # 1. First Step Retrieval (Revised Logic with s0 Bonus)
    logging.info("Filtering nodes and calculating scores for the first step...")

    # Pre-filter nodes based on metadata
    pre_filtered_nodes = {}
    start_time = time.perf_counter()
    for node_id, data in G.nodes(data=True):
        # Metadata checks
        node_knowledge_tags_str = data.get("knowledge_tag", "")
        tag_match = True
        if query_knowledge_tags:
            node_knowledge_tags = [
                tag.strip() for tag in node_knowledge_tags_str.split(",") if tag.strip()
            ]
            tag_match = any(
                query_tag in node_knowledge_tags for query_tag in query_knowledge_tags
            )
        if not tag_match:
            continue

        pre_filtered_nodes[node_id] = G.nodes[node_id].get("text", "")

    if not pre_filtered_nodes:
        logging.warning("No candidate nodes found matching the query tags/type.")
        return [], time.perf_counter() - start_time
    logging.info(
        f"Found {len(pre_filtered_nodes)} initial candidate nodes based on metadata."
    )

    candidate_node_ids_for_sim = list(pre_filtered_nodes.keys())
    candidate_texts = [
        pre_filtered_nodes[node_id] for node_id in candidate_node_ids_for_sim
    ]

    if not candidate_node_ids_for_sim:
        logging.warning("No candidates remaining for similarity calculation.")
        return [], time.perf_counter() - start_time

    step_similarities = {}
    for i in range(0, len(candidate_texts), batch_size):
        batch_texts = candidate_texts[i : i + batch_size]
        batch_node_ids = candidate_node_ids_for_sim[i : i + batch_size]

        queries = [
            query_template.format(
                prefix=RERANKER_PREFIX,
                instruction=reranker_instruction,
                query=query_text,
            )
        ] * len(batch_texts)
        documents = [
            document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
            for doc in batch_texts
        ]

        try:
            response = session.post(
                reranker_url,
                json={
                    "model": reranker_model_name,
                    "text_1": queries,
                    "text_2": documents,
                    "truncate_prompt_tokens": -1,
                },
            )
            response_data = response.json()
            score_list = response_data.get("data", [])

            for item in score_list:
                original_node_id = batch_node_ids[item["index"]]
                step_similarities[original_node_id] = item["score"]
        except Exception as e:
            logging.error(f"Initial reranker API call failed: {e}", exc_info=True)
            # Continue with potentially fewer scores
            continue

    # Calculate combined score with s0 bonus and find the best node
    best_combined_score = -np.inf
    start_node = None
    start_node_similarity = -1.0  # Store raw step similarity for logging

    for node_id in candidate_node_ids_for_sim:
        step_sim = step_similarities.get(node_id, -1.0)
        is_s0_bonus = 1.0 if node_id.endswith("_s0") else 0.0
        combined_score = (
            FIRST_STEP_WEIGHT_QUERY_SIM * step_sim
            + FIRST_STEP_WEIGHT_S0_BONUS * is_s0_bonus
        )
        logging.debug(
            f"  Candidate: {node_id}, StepSim: {step_sim:.4f}, Is_s0: {is_s0_bonus:.1f}, Combined: {combined_score:.4f}"
        )

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            start_node = node_id
            start_node_similarity = step_sim

    if start_node is None:
        logging.error("Could not determine a start node after scoring.")
        return [], 0

    logging.info(
        f"Selected start node: '{start_node}' with Combined Score {best_combined_score:.4f} (StepSim: {start_node_similarity:.4f})"
    )

    # 2. Graph Traversal with Beam Search
    beam = [([start_node], best_combined_score)]
    completed_paths = []

    logging.info(
        f"Starting graph traversal with beam search (width={beam_width}, 0.5 QuerySim + 0.5 FlowScore)..."
    )
    while beam:
        all_next_candidates = []
        paths_to_remove_from_beam = []

        rerank_candidates = []
        rerank_docs = []

        for i, (path, path_score) in enumerate(beam):
            current_node = path[-1]

            if len(path) >= MAX_TEMPLATE_LENGTH:
                completed_paths.append((path, path_score))
                paths_to_remove_from_beam.append(i)
                continue

            successors = [
                node for node in G.successors(current_node) if node not in path
            ]

            if not successors:
                completed_paths.append((path, path_score))
                paths_to_remove_from_beam.append(i)
                continue

            current_path_text = "\n".join([node_texts.get(n, "") for n in path])
            for successor_node in successors:
                successor_text = node_texts.get(successor_node, "")
                full_doc_text = f"{current_path_text}\n{successor_text}"
                rerank_candidates.append((i, successor_node))
                rerank_docs.append(full_doc_text)

        query_similarities = {}  # Key: (path_index, successor_node), Value: score
        if rerank_docs:
            for i in range(0, len(rerank_docs), batch_size):
                batch_docs = rerank_docs[i : i + batch_size]
                batch_candidates = rerank_candidates[i : i + batch_size]
                queries = [
                    query_template.format(
                        prefix=RERANKER_PREFIX,
                        instruction=reranker_instruction,
                        query=query_text,
                    )
                ] * len(batch_docs)
                documents = [
                    document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
                    for doc in batch_docs
                ]
                try:
                    response = session.post(
                        reranker_url,
                        json={
                            "model": reranker_model_name,
                            "text_1": queries,
                            "text_2": documents,
                            "truncate_prompt_tokens": -1,
                        },
                    ).json()
                    for item in response.get("data", []):
                        original_candidate = batch_candidates[item["index"]]
                        query_similarities[original_candidate] = item["score"]
                except Exception as e:
                    logging.error(
                        f"Traversal reranker API call failed: {e}", exc_info=True
                    )
                    continue

        for i, (path, path_score) in enumerate(beam):
            if i in paths_to_remove_from_beam:
                continue

            current_node = path[-1]
            successors = [
                node for node in G.successors(current_node) if node not in path
            ]
            sequential_child_node = next(
                (
                    target
                    for _, target, data in G.out_edges(current_node, data=True)
                    if data.get("edge_type") == "sequential"
                ),
                None,
            )

            for candidate_node in successors:
                query_sim = query_similarities.get((i, candidate_node), -1.0)
                if query_sim < SIMILARITY_THRESHOLD_RERANK:
                    continue

                flow_score = 1.0 if candidate_node == sequential_child_node else 0.0
                combined_score = 0.5 * query_sim + 0.5 * flow_score
                new_path = path + [candidate_node]
                new_score = combined_score
                all_next_candidates.append((new_path, new_score))

        for index in sorted(paths_to_remove_from_beam, reverse=True):
            if index < len(beam):
                del beam[index]

        if not all_next_candidates:
            completed_paths.extend(beam)
            break

        all_next_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beam = all_next_candidates[:beam_width]

    if not completed_paths and beam:
        completed_paths.extend(beam)

    if not completed_paths:
        logging.warning("No paths found during traversal. Returning start node only.")
        dynamic_template_nodes = [start_node]
    else:
        completed_paths.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        best_path, best_score = completed_paths[0]
        dynamic_template_nodes = best_path
        logging.info(
            f"Best path found with normalized score {best_score/len(best_path):.4f} and length {len(dynamic_template_nodes)}"
        )

    # 3. Assemble Template
    final_template = [
        node_texts.get(node_id, f"Error: Text not found for {node_id}")
        for node_id in dynamic_template_nodes
    ]
    logging.info(f"Generated template with {len(final_template)} steps.")
    logging.info(f"Query text: {query_text}")
    logging.info(f"Template nodes: {final_template}")

    return final_template, time.perf_counter() - start_time


def generate_dynamic_template_rerank_beam_search(
    query_text: str,
    query_knowledge_tags: list[str],
    G: nx.Graph,
    node_texts,
    reranker_url: str,
    reranker_model_name: str,
    reranker_instruction: str,
    batch_size: int = 128,
    beam_width: int = 5,
):
    """
    Generates a dynamic problem-solving template using beam search initialized with the top 5 candidate nodes.
    """
    RERANKER_PREFIX = r'<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    RERANKER_SUFFIX = r"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    logging.info(f"Received query: '{query_text}'")
    logging.info(f"Query knowledge tags: {query_knowledge_tags}")

    if not query_text:
        logging.warning("Query text is empty. Cannot generate template.")
        return [], 0

    query_template = r"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = r"<Document>: {doc}{suffix}"
    session = requests.Session()
    start_time = time.perf_counter()

    # 1. First Step Retrieval
    logging.info("Filtering nodes for the first step...")
    pre_filtered_nodes = {}
    for node_id, data in G.nodes(data=True):
        node_knowledge_tags_str = data.get("knowledge_tag", "")
        tag_match = True
        if query_knowledge_tags:
            node_knowledge_tags = [
                tag.strip() for tag in node_knowledge_tags_str.split(",") if tag.strip()
            ]
            tag_match = any(
                query_tag in node_knowledge_tags for query_tag in query_knowledge_tags
            )
        if tag_match:
            pre_filtered_nodes[node_id] = G.nodes[node_id].get("text", "")

    if not pre_filtered_nodes:
        logging.warning("No candidate nodes found matching the query tags.")
        return [], time.perf_counter() - start_time

    candidate_node_ids = list(pre_filtered_nodes.keys())
    candidate_texts = [pre_filtered_nodes[node_id] for node_id in candidate_node_ids]

    # Batch rerank initial candidates
    step_similarities = {}

    for i in range(0, len(candidate_texts), batch_size):
        batch_texts = candidate_texts[i : i + batch_size]
        batch_node_ids = candidate_node_ids[i : i + batch_size]
        queries = [
            query_template.format(
                prefix=RERANKER_PREFIX,
                instruction=reranker_instruction,
                query=query_text,
            )
        ] * len(batch_texts)
        documents = [
            document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
            for doc in batch_texts
        ]
        try:
            response = session.post(
                reranker_url,
                json={
                    "model": reranker_model_name,
                    "text_1": queries,
                    "text_2": documents,
                    "truncate_prompt_tokens": -1,
                },
            ).json()
            for item in response.get("data", []):
                original_node_id = batch_node_ids[item["index"]]
                step_similarities[original_node_id] = item["score"]
        except Exception as e:
            logging.error(f"Reranker API call failed: {e}", exc_info=True)
            continue

    # Calculate combined scores and identify top candidates for beam search
    scored_candidates = []
    for node_id in candidate_node_ids:
        step_sim = step_similarities.get(node_id, -1.0)
        is_s0_bonus = 1.0 if node_id.endswith("_s0") else 0.0
        combined_score = (
            FIRST_STEP_WEIGHT_QUERY_SIM * step_sim
            + FIRST_STEP_WEIGHT_S0_BONUS * is_s0_bonus
        )
        scored_candidates.append((node_id, combined_score, step_sim))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # 2. Graph Traversal with Beam Search

    beam = []
    for node_id, combined_score, step_sim in scored_candidates[:beam_width]:
        if step_sim >= SIMILARITY_THRESHOLD_RERANK:
            beam.append(([node_id], combined_score))
            logging.info(
                f"Initializing beam with node '{node_id}' (Score: {combined_score:.4f}, Sim: {step_sim:.4f})"
            )

    if not beam:
        logging.warning(
            f"No initial nodes met the similarity threshold of {SIMILARITY_THRESHOLD_RERANK}. Cannot start beam search."
        )
        return [], time.perf_counter() - start_time

    completed_paths = []
    logging.info(f"Starting graph traversal with beam search (width={beam_width})...")

    while beam:
        all_next_candidates = []
        paths_to_remove_from_beam = []

        # Collect all potential next paths from all current beams to rerank in a single batch
        rerank_candidates = []  # List of tuples: (path_index, successor_node)
        rerank_docs = []

        for i, (path, path_score) in enumerate(beam):
            current_node = path[-1]

            if len(path) >= MAX_TEMPLATE_LENGTH:
                completed_paths.append((path, path_score))
                paths_to_remove_from_beam.append(i)
                continue

            successors = [
                node for node in G.successors(current_node) if node not in path
            ]

            if not successors:
                completed_paths.append((path, path_score))
                paths_to_remove_from_beam.append(i)
                continue

            # For each successor, create the full path text for reranking
            current_path_text = "\n".join([node_texts.get(n, "") for n in path])
            for successor_node in successors:
                successor_text = node_texts.get(successor_node, "")
                # The document for the reranker is the concatenation of the path so far and the next step
                full_doc_text = f"{current_path_text}\n{successor_text}"
                rerank_candidates.append((i, successor_node))
                rerank_docs.append(full_doc_text)

        # Rerank all collected candidate paths in batches
        query_similarities = {}  # Key: (path_index, successor_node), Value: score
        if rerank_docs:
            for i in range(0, len(rerank_docs), batch_size):
                batch_docs = rerank_docs[i : i + batch_size]
                batch_candidates = rerank_candidates[i : i + batch_size]

                queries = [
                    query_template.format(
                        prefix=RERANKER_PREFIX,
                        instruction=reranker_instruction,
                        query=query_text,
                    )
                ] * len(batch_docs)
                documents = [
                    document_template.format(doc=doc, suffix=RERANKER_SUFFIX)
                    for doc in batch_docs
                ]

                try:
                    response = session.post(
                        reranker_url,
                        json={
                            "model": reranker_model_name,
                            "text_1": queries,
                            "text_2": documents,
                            "truncate_prompt_tokens": -1,
                        },
                    ).json()
                    for item in response.get("data", []):
                        # Map score back to its original path and successor
                        original_candidate = batch_candidates[item["index"]]
                        query_similarities[original_candidate] = item["score"]
                except Exception as e:
                    logging.error(
                        f"Traversal reranker API call failed: {e}", exc_info=True
                    )
                    # We can decide to continue without these scores or handle error differently
                    continue

        # Now, build the next generation of beams using the calculated scores
        for i, (path, path_score) in enumerate(beam):
            if i in paths_to_remove_from_beam:
                continue

            current_node = path[-1]
            successors = [
                node for node in G.successors(current_node) if node not in path
            ]
            sequential_child_node = next(
                (
                    target
                    for _, target, data in G.out_edges(current_node, data=True)
                    if data.get("edge_type") == "sequential"
                ),
                None,
            )

            for candidate_node in successors:
                # Get the score for the full path text
                query_sim = query_similarities.get((i, candidate_node), -1.0)

                if query_sim < SIMILARITY_THRESHOLD_RERANK:
                    continue

                flow_score = 1.0 if candidate_node == sequential_child_node else 0.0
                # The score for this step is a combination of the full-path rerank score and the flow score
                combined_score = 0.5 * query_sim + 0.5 * flow_score
                new_path = path + [candidate_node]
                # The new path's total score is the sum of scores from previous steps
                new_score = combined_score
                all_next_candidates.append((new_path, new_score))

        # Update beam for the next iteration
        # First, remove paths that have been completed or had errors
        for index in sorted(paths_to_remove_from_beam, reverse=True):
            # Check if the beam has been modified by error handling
            if index < len(beam):
                del beam[index]

        if not all_next_candidates:
            # If no new candidates were generated, add remaining beams to completed and break
            completed_paths.extend(beam)
            break

        # Normalize score by path length to avoid favoring shorter paths, and select the best
        all_next_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
        beam = all_next_candidates[:beam_width]

    if not completed_paths and beam:
        completed_paths.extend(beam)

    if not completed_paths:
        logging.warning("No paths found during beam search.")
        return [], time.perf_counter() - start_time

    # Select the best path based on the highest normalized score
    completed_paths.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
    best_path, best_score = completed_paths[0]
    dynamic_template_nodes = best_path
    logging.info(
        f"Best path found with normalized score {best_score/len(best_path):.4f} and length {len(dynamic_template_nodes)}"
    )

    # 3. Assemble Template
    final_template = [
        node_texts.get(node_id, f"Error: Text not found for {node_id}")
        for node_id in dynamic_template_nodes
    ]
    logging.info(f"Generated template with {len(final_template)} steps.")

    return final_template, time.perf_counter() - start_time


def precompute_node_text(G: nx.Graph):
    node_texts = {}

    try:
        logging.info("Precomputing text for all graph nodes...")
        all_node_ids = list(G.nodes())
        all_node_data = [G.nodes[node_id] for node_id in all_node_ids]
        all_texts = [data.get("text", "") for data in all_node_data]

        node_texts_to_encode = [
            (node_id, text) for node_id, text in zip(all_node_ids, all_texts) if text
        ]
        texts_for_encoding = [item[1] for item in node_texts_to_encode]

        if texts_for_encoding:
            logging.info(f"Encoding {len(texts_for_encoding)} non-empty node texts...")

        node_id_to_encode_index = {
            item[0]: i for i, item in enumerate(node_texts_to_encode)
        }

        for node_id in all_node_ids:
            text = G.nodes[node_id].get("text", "")
            node_texts[node_id] = text

        logging.info(f"Text computed for {len(node_texts_to_encode)} nodes.")
    except Exception as e:
        logging.error(f"Error computing embeddings: {e}", exc_info=True)
        exit()
    return node_texts, node_id_to_encode_index
