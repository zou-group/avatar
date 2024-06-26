import torch
import os.path as osp

from avatar.utils.format import format_checked
from avatar.utils.topk import get_top_k_indices 
from avatar.tools.tool import Tool
from stark_qa.tools.process_text import chunk_text
from stark_qa.tools.api import get_openai_embedding, get_openai_embeddings


class GetFullInfo(Tool):
    """
    A class to get the complete textual and relational information of a node.

    Args:
        kb: The knowledge base containing the node information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
    
    @format_checked
    def __call__(self, node_id: int) -> str:
        """
        Get the complete textual and relational information of a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            str: The complete textual and relational information of the node.
        """
        return self.kb.get_doc_info(node_id, add_rel=True, compact=False)
        
    def __str__(self):
        return 'get_full_info(node_id: int) -> full_info: str'
    
    def __repr__(self):
        return 'Return a string containing the complete textual and relational information of the node with ID `node_id`.' 


class GetTextInfo(Tool):
    """
    A class to get the textual information of a node.

    Args:
        kb: The knowledge base containing the node information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"

    @format_checked
    def __call__(self, node_id: int) -> str:
        """
        Get the textual information of a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            str: The textual information of the node.
        """
        return self.kb.get_doc_info(node_id, add_rel=False, compact=False)
            
    def __str__(self):
        return 'get_text_info(node_id: int) -> text_info: str'
    
    def __repr__(self):
        return 'Return a string containing the textual information of the node with ID `node_id`.' 


class GetRelationInfo(Tool):
    """
    A class to get the one-hop relational information of a node.

    Args:
        kb: The knowledge base containing the node information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_rel_info'), "kb must have a method 'get_rel_info'"

    @format_checked
    def __call__(self, node_id: int) -> str:
        """
        Get the one-hop relational information of a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            str: The one-hop relational information of the node.
        """
        return self.kb.get_rel_info(node_id)
            
    def __str__(self):
        return 'get_relation_info(node_id: int) -> relation_info: str'
    
    def __repr__(self):
        return (
            "Return a string containing the one-hop relational information of the node with ID `node_id`. For example, get_relation_info(1024) returns a string "
            "'- relations: - <one-hop relation type 1>: <names of the nodes connected with 1024 via relation 1> ...'. Note that some nodes may have no relations."
        )


class GetRelevantChunk(Tool): 
    """
    A class to extract relevant chunks of information related to a specific attribute from a node.

    Args:
        kb: The knowledge base containing the node information.
        chunk_emb_dir: Directory to save/load chunk embeddings.
        emb_model (str): The name of the model to use for generating embeddings.
    """

    def __init__(self, 
                 kb, 
                 chunk_emb_dir: str,
                 emb_model: str = 'text-embedding-ada-002',
                 **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        self.emb_model = emb_model
        self.chunk_emb_dir = chunk_emb_dir

    @format_checked
    def __call__(self, 
                 node_id: int, 
                 attribute: str, 
                 k: int = 3, 
                 chunk_size: int = 256, 
                 threshold: float = 0.80) -> str:
        """
        Extract and concatenate at most `k` chunks related to an attribute for a node.

        Args:
            node_id (int): The ID of the node.
            attribute (str): The attribute to find relevant chunks for.
            k (int, optional): The maximum number of chunks to extract. Default is 3.
            chunk_size (int, optional): The size of each chunk. Default is 256.
            threshold (float, optional): The similarity threshold to select chunks. Default is 0.80.

        Returns:
            str: The concatenated relevant chunks.
        """
        chunk = ''
        if hasattr(self.kb[node_id], attribute):
            try:
                chunk = getattr(self.kb[node_id], attribute)
                if isinstance(chunk, list):
                    chunk = ';\n'.join(chunk)
                else:
                    chunk = str(chunk)
            except:
                pass
        
        if len(chunk) == 0:
            doc = self.kb.get_doc_info(node_id, add_rel=True, compact=True)
            chunks = chunk_text(doc, chunk_size=chunk_size)
            chunk_path = osp.join(self.chunk_emb_dir, f'{node_id}_size={chunk_size}.pt')
            if osp.exists(chunk_path):
                chunk_embs = torch.load(chunk_path)
            else:
                chunk_embs = get_openai_embeddings(chunks, model=self.emb_model)
                torch.save(chunk_embs, chunk_path)
            attribute_emb = get_openai_embedding(attribute, model=self.emb_model)
            sel_ids, similarity = get_top_k_indices(attribute_emb, chunk_embs, return_similarity=True)
            
            if sum(similarity > threshold) > 0:
                num = min(k, sum(similarity > threshold))
                sel_ids = torch.LongTensor(sel_ids)[:num].tolist()
                chunk = ';\n'.join([chunks[idx] for idx in sel_ids])
            else:
                chunk = chunks[sel_ids[0]]
        return chunk
    
    def __str__(self):
        return 'get_relevant_chunk(node_id: int, attribute: str, k: int = 3, chunk_size: int = 256, threshold: float = 0.80) -> chunk: str'
    
    def __repr__(self):
        return (
            f"Extracts and concatenates at most `k` chunks (k=3 by default) related to `attribute` for `node_id`. Each chunk has a size of `chunk_size` "
            "and its similarity with `query` will be no less than `threshold` (chunk_size=256 and threshold=0.80 by default). "
            "For example, get_relevant_chunk(node_id=1024, attribute='user level') could return a string containing relevant information about the user level from node 1024."
        )
