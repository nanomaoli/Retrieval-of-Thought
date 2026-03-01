import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SIMILARITY_THRESHOLD = 0.7  # Threshold for query similarity during traversal
MAX_TEMPLATE_LENGTH = 8
FIRST_STEP_WEIGHT_QUERY_SIM = 0.8  # Weight for query similarity in first step selection
FIRST_STEP_WEIGHT_S0_BONUS = 0.2  # Weight for being the first step (_s0)

def generate_dynamic_template(query_text, query_knowledge_tags, query_template_type, G, node_embeddings_cache, embedding_model,):
    """
    Generates a dynamic problem-solving template based on the query using s0 bonus for first step.
    """

    logging.info(f"Received query: '{query_text}'")
    logging.info(f"Query knowledge tags: {query_knowledge_tags}")
    logging.info(f"Query template type: {query_template_type}")

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
    for node_id, data in G.nodes(data=True):
        # Metadata checks
        node_template_type = data.get("template_type", "")
        node_knowledge_tags_str = data.get("knowledge_tag", "")
        if node_template_type != query_template_type:
            continue
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
        return []
    logging.info(
        f"Found {len(pre_filtered_nodes)} initial candidate nodes based on metadata."
    )

    # Prepare data for batch similarity calculation
    candidate_step_embeddings = []
    candidate_node_ids_for_sim = []
    for node_id in pre_filtered_nodes:
        step_emb = node_embeddings_cache.get(node_id)
        if isinstance(step_emb, np.ndarray):  # Ensure it's a valid embedding
            candidate_step_embeddings.append(step_emb)
            candidate_node_ids_for_sim.append(node_id)
        else:
            logging.warning(
                f"Node {node_id} has invalid embedding type ({type(step_emb)}), skipping similarity."
            )

    if not candidate_node_ids_for_sim:
        logging.warning("No candidates remaining for similarity calculation.")
        return []

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
        return []

    # Calculate combined score with s0 bonus and find the best node
    best_combined_score = -np.inf
    start_node = None
    start_node_similarity = -1.0  # Store raw step similarity for logging

    for i, node_id in enumerate(candidate_node_ids_for_sim):
        # Add boundary check for safety
        step_sim = step_similarities[i] if i < len(step_similarities) else -1.0

        # Ensure sim is a valid float
        if not isinstance(step_sim, (float, np.float32, np.float64)):
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
        return []

    logging.info(
        f"Selected start node: '{start_node}' with Combined Score {best_combined_score:.4f} (StepSim: {start_node_similarity:.4f})"
    )

    # 2. Graph Traversal (Hybrid Logic - 0.5 query sim / 0.5 flow score)
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

        if best_node_query_similarity < SIMILARITY_THRESHOLD:
            logging.info(
                f"Termination: Best next node's query similarity ({best_node_query_similarity:.4f}) is below threshold ({SIMILARITY_THRESHOLD})."
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

    return final_template
