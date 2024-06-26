import os.path as osp
import torch
from typing import List, Union
from tqdm import tqdm

from avatar.utils.format import format_checked
from avatar.tools.tool import Tool
from stark_qa.tools.api import get_openai_embedding, get_openai_embeddings


class GetNodeEmbedding(Tool):
    """
    A class to get embeddings for nodes specified by their IDs.
    
    Args:
        kb: The knowledge base containing node information.
        node_emb_dir (str): The directory to save or load node embeddings.
        emb_model (str): The name of the model to use for generating embeddings.
    """

    def __init__(self, kb, 
                 node_emb_dir: str, 
                 emb_model: str = 'text-embedding-ada-002',
                 **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)
        self.kb = kb
        self.node_emb_dir = node_emb_dir
        self.emb_model = emb_model
        self.nodes_emb_path = osp.join(node_emb_dir, 'all_node_emb_dict.pt')
        self.candidate_emb_path = osp.join(node_emb_dir, 'candidate_emb_dict.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if osp.exists(self.nodes_emb_path):
            self.node_emb_dict = torch.load(self.nodes_emb_path)
            self.node_embs = torch.cat([self.node_emb_dict[i] for i in range(len(self.node_emb_dict))], dim=0)
        else:
            if osp.exists(self.candidate_emb_path):
                self.node_emb_dict = torch.load(self.candidate_emb_path)
        self.candidate_ids = self.kb.candidate_ids

    @format_checked
    def __call__(self, node_ids: Union[int, List[int]]) -> torch.Tensor:
        """
        Get embeddings for the specified node IDs.
        
        Args:
            node_ids (Union[int, List[int]]): A single node ID or a list of node IDs.
            
        Returns:
            torch.Tensor: A tensor of embeddings for the specified nodes.
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        embs = []
        print(f'get_node_embedding - input node_ids {len(node_ids)}')
        if osp.exists(self.nodes_emb_path):
            return self.node_embs[node_ids]
        for node_id in node_ids:
            emb_path = osp.join(self.node_emb_dir, f'{node_id}.pt')
            if node_id in self.candidate_ids:
                print('get_node_embedding - load from candidate_ids')
                emb = self.node_emb_dict[node_id]
            elif osp.exists(emb_path):
                print(f'get_node_embedding - load from {emb_path}')
                emb = torch.load(emb_path)
            else:
                print(f'get_node_embedding - compute embedding and save to {emb_path}')
                emb = get_openai_embedding(self.kb.get_doc_info(node_id, add_rel=True, compact=True), model=self.emb_model)
                torch.save(emb, emb_path)
            embs.append(emb)
        return torch.cat(embs, dim=0).view(len(node_ids), -1)

    def __str__(self):
        return 'get_node_embedding(node_ids: Union[int, List[int]]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Get embeddings for nodes specified by `node_ids`. The result is a tensor of size (len(node_ids), hidden_dim). "
                "For example, get_node_embedding([12, 34]) returns a tensor of size (2, hidden_dim), while get_node_embedding(12) "
                "returns a tensor of size (1, hidden_dim). For efficiency, use this function with multiple node IDs at once, "
                "instead of calling it separately for each ID.")


class GetTextEmbedding(Tool):
    """
    A class to get embeddings for text strings.
    
    Args:
        emb_model (str): The name of the model to use for generating embeddings.
    """

    def __init__(self, emb_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__()
        self.emb_model = emb_model

    @format_checked
    def __call__(self, string: Union[str, List[str]]) -> torch.Tensor:
        """
        Get embeddings for the specified text strings.
        
        Args:
            string (Union[str, List[str]]): A single string or a list of strings.
            
        Returns:
            torch.Tensor: A tensor of embeddings for the specified strings.
        """
        if isinstance(string, str):
            string = [string]
        assert all([len(s) > 0 for s in string]), 'every string in the list to be embedded should be non-empty'
        embs = get_openai_embeddings(string, model=self.emb_model)
        print(f'get_text_embedding - input {string} - output shape {embs.size()}')
        return embs

    def __str__(self):
        return 'get_text_embedding(string: Union[str, List[str]]) -> embedding: torch.Tensor'

    def __repr__(self):
        return ("Embed a string or a list of N strings into a tensor of size (N, hidden_dim). For efficiency, include multiple strings in the list at once, "
                "rather than calling the function separately for each string.")


class ComputeSimilarity(Tool):
    """
    A class to compute similarity between a query and node information.
    
    Args:
        kb: The knowledge base containing node information.
        chunk_size (int): The size of chunks for processing.
        chunk_emb_dir (str): The directory to save or load chunk embeddings.
        node_emb_dir (str): The directory to save or load node embeddings.
        emb_model (str): The name of the model to use for generating embeddings.
    """

    def __init__(self, kb, 
                 chunk_size: int, 
                 chunk_emb_dir: str, 
                 node_emb_dir: str, 
                 emb_model: str = 'text-embedding-ada-002',
                 **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)
        self.node_emb_dir = node_emb_dir
        self.emb_model = emb_model
        candidate_emb_path = osp.join(node_emb_dir, 'candidate_emb_dict.pt')
        self.candidate_emb_dict = torch.load(candidate_emb_path)
        self.candidate_ids = self.kb.candidate_ids
        self.get_emb = GetNodeEmbedding(kb, node_emb_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @format_checked
    def __call__(self, query: str, node_ids: List[int]) -> List[float]:
        """
        Compute similarity between a query and node information.
        
        Args:
            query (str): The query string.
            node_ids (List[int]): A list of node IDs to compare against.
            
        Returns:
            List[float]: A list of similarity scores.
        """
        print(f'compute_similarity - input query {query}, node_ids {len(node_ids)}')

        if len(node_ids) == 0:
            return []
        query_emb = get_openai_embedding(query, model=self.emb_model).to(self.device)
        print(f'compute_similarity - query emb {query_emb.size()}')

        embs = []
        for node_id in node_ids:
            embs.append(self.get_emb(node_id))
        embs = torch.cat(embs, dim=0).to(self.device)
        sim = torch.matmul(query_emb, embs.T).view(-1)
        print(f'compute_similarity - sim {sim.size()}')
        return sim.cpu().tolist()

    def __str__(self):
        return 'compute_similarity(query: str, node_ids: List[int]) -> node_scores: List[float]'

    def __repr__(self):
        return ("Compute the similarity between a `query` and the information of each node in `node_ids`, based on their embedding similarity. "
                "For example, compute_similarity(query='Dress suitable as a gift.', node_ids=[12, 34]) returns a list of similarity scores, e.g., [0.56, -0.12]. "
                "Here `query` can be described in flexible terms or sentences. For efficiency, use this function with multiple node IDs at once, "
                "instead of calling it separately for each node ID.")


class ComputeQueryNodeSimilarity(Tool):
    """
    A class to compute similarity between a query and nodes based on their embeddings.
    
    Args:
        kb: The knowledge base containing node information.
        chunk_size (int): The size of chunks for processing.
        chunk_emb_dir (str): The directory to save or load chunk embeddings.
        node_emb_dir (str): The directory to save or load node embeddings.
        emb_model (str): The name of the model to use for generating embeddings.
    """

    def __init__(self, kb, 
                 chunk_size: int, 
                 chunk_emb_dir: str, 
                 node_emb_dir: str, 
                 emb_model: str = 'text-embedding-ada-002',
                 **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)
        self.emb_model = emb_model
        self.node_emb_dir = node_emb_dir
        candidate_emb_path = osp.join(node_emb_dir, 'candidate_emb_dict.pt')
        self.candidate_emb_dict = torch.load(candidate_emb_path)
        self.candidate_ids = self.kb.candidate_ids
        self.get_emb = GetNodeEmbedding(kb, node_emb_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @format_checked
    def __call__(self, query: str, node_ids: List[int]) -> torch.Tensor:
        """
        Compute similarity between a query and nodes based on their embeddings.
        
        Args:
            query (str): The query string.
            node_ids (List[int]): A list of node IDs to compare against.
            
        Returns:
            torch.Tensor: A tensor of similarity scores.
        """
        print(f'compute_query_node_similarity - input query {query}, node_ids {len(node_ids)}')
        query_emb = get_openai_embedding(query, model=self.emb_model).to(self.device)
        print(f'compute_query_node_similarity - query emb {query_emb.size()}')

        embs = []
        for node_id in node_ids:
            embs.append(self.get_emb(node_id))
        embs = torch.cat(embs, dim=0).to(self.device)
        sim = torch.matmul(query_emb, embs.T).view(-1)
        print(f'compute_query_node_similarity - sim {sim.size()}')
        return sim.cpu()

    def __str__(self):
        return 'compute_query_node_similarity(query: str, node_ids: List[int]) -> similarity: torch.Tensor'

    def __repr__(self):
        return ("Compute the similarity between a `query` and nodes specified by `node_ids`, based on their embedding similarity. "
                "For example, compute_query_node_similarity(query='Dress suitable as a gift.', node_ids=[12, 34]) returns a list of similarity scores, "
                "e.g., torch.Tensor([0.56, -0.12]), for nodes with IDs 12 and 34. For efficiency, use this function with multiple node IDs at once, "
                "instead of calling it separately for each node ID.")


class ComputeCosineSimilarity(Tool):
    """
    A class to compute pair-wise cosine similarity between two embedding matrices.
    
    Args:
        kb: The knowledge base containing node information.
    """

    def __init__(self, kb, **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @format_checked
    def __call__(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        """
        Compute pair-wise cosine similarity between two embedding matrices.
        
        Args:
            embedding_1 (torch.Tensor): The first embedding matrix.
            embedding_2 (torch.Tensor): The second embedding matrix.
            
        Returns:
            torch.Tensor: A tensor of similarity scores.
        """
        print(f'compute_cosine_similarity - input emb1 {embedding_1.size()}, emb2 {embedding_2.size()}')
        hidden_dim = embedding_1.size(1) if len(embedding_1.size()) > 1 else embedding_1.size(0)
        emb_1 = embedding_1.view(-1, hidden_dim).to(self.device)
        emb_2 = embedding_2.view(-1, hidden_dim).to(self.device)
        similarity = torch.matmul(emb_1, emb_2.T)
        similarity = similarity / torch.norm(emb_1, dim=1, keepdim=True)
        similarity = similarity / torch.norm(emb_2, dim=1, keepdim=True).T
        return similarity.cpu()

    def __str__(self):
        return 'compute_cosine_similarity(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> similarity: torch.Tensor'

    def __repr__(self):
        return ("Compute pair-wise cosine similarity between two embedding matrices. If the size of `embedding_1` is (n, d), and the size of `embedding_2` is (m, d), "
                "then the size of the resulting tensor `similarity` is (n, m). For example, compute_cosine_similarity(embedding_1=torch.randn(2, 768), "
                "embedding_2=torch.randn(3, 768)) returns a tensor of size (2, 3), where the (i,j) element is the cosine similarity between the i-th row of "
                "`embedding_1` and the j-th row of `embedding_2`.")
