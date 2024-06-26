import torch


def get_top_k_indices(emb: torch.FloatTensor, 
                      candidate_embs: torch.FloatTensor, 
                      k: int = -1, 
                      return_similarity: bool = False) -> list:
    """
    Get the top-k indices of candidates based on the similarity to the given embedding.

    Args:
        emb (torch.FloatTensor): Embedding of the query.
        candidate_embs (torch.FloatTensor): Embeddings of the candidates.
        k (int, optional): Number of top candidates to return. If k <= 0, rank all candidates. Default is -1.
        return_similarity (bool, optional): Whether to return the similarities along with the indices. Default is False.

    Returns:
        list: List of top-k indices. If return_similarity is True, also returns the list of similarities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = emb.to(device)
    candidate_embs = candidate_embs.to(device)

    # Compute similarity
    sim = torch.matmul(emb, candidate_embs.T).cpu().view(-1)

    if k > 0:
        # Get the top-k indices
        indices = torch.topk(sim, k=min(k, len(sim)), dim=-1, sorted=True).indices.view(-1).tolist()
    else:
        # Get all indices sorted by similarity
        indices = torch.argsort(sim, dim=-1, descending=True).view(-1).tolist()

    if return_similarity:
        return indices, sim[indices].tolist()
    return indices
