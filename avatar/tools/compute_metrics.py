from typing import List
from avatar.utils.format import format_checked
from avatar.tools.tool import Tool
from stark_qa.tools.process_text import exact_match_score, recall_score, f1_score


class ComputeF1Score(Tool): 
    """
    Class to compute F1 score for a given string against a list of nodes in the knowledge base.
    
    Args:
        kb: The knowledge base containing the nodes.
        **kwargs: Additional arguments.
    """

    def __init__(self, kb, **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, string: str, node_ids: List[int]) -> List[float]:
        """
        Compute F1 scores between the input string and the full information of nodes in the knowledge base.

        Args:
            string (str): The input string to compare.
            node_ids (List[int]): List of node IDs in the knowledge base.

        Returns:
            List[float]: List of F1 scores for each node.
        """
        docs = [self.kb.get_doc_info(node_id, add_rel=False, compact=True) for node_id in node_ids]
        return [f1_score(string.lower(), doc.lower()) for doc in docs]

    def __str__(self):
        return 'compute_f1_score(string: str, node_ids: List[int]) -> f1_match_score: List[float]'

    def __repr__(self):
        return ("For each node in `node_ids`, this function computes F1 scores between `string` and the full information of the node. "
                "For example, compute_f1_score(string='Adidas', node_ids=[2000, 3000]) returns a list of F1 scores, e.g., [0.05, 1.0], "
                "which represent the F1 scores between 'Adidas' and the full information of brand nodes with IDs 2000 and 3000, respectively. "
                "This function provides a more flexible matching metric than the exact match score.")


class ComputeRecallScore(Tool): 
    """
    Class to compute recall score for a given string against a list of nodes in the knowledge base.
    
    Args:
        kb: The knowledge base containing the nodes.
        **kwargs: Additional arguments.
    """

    def __init__(self, kb, **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, string: str, node_ids: List[int]) -> List[float]:
        """
        Compute recall scores between the input string and the full information of nodes in the knowledge base.

        Args:
            string (str): The input string to compare.
            node_ids (List[int]): List of node IDs in the knowledge base.

        Returns:
            List[float]: List of recall scores for each node.
        """
        docs = [self.kb.get_doc_info(node_id, add_rel=False, compact=True) for node_id in node_ids]
        return [recall_score(string.lower(), doc.lower()) for doc in docs]

    def __str__(self):
        return 'compute_recall_score(string: str, node_ids: List[int]) -> recall: List[float]'

    def __repr__(self):
        return ("For each node in `node_ids`, this function computes recall scores between `string` and the full information of the node. "
                "For example, compute_recall_score(string='H&M', node_ids=[2000, 3000]) returns a list of recall scores, e.g., [0.33, 1.0], "
                "which represent the recall scores between 'H&M' and the full information of brand nodes with IDs 2000 and 3000, respectively. "
                "This function is a more flexible matching metric than the exact match score.")


class ComputeExactMatchScore(Tool): 
    """
    Class to compute exact match score for a given string against a list of nodes in the knowledge base.
    
    Args:
        kb: The knowledge base containing the nodes.
        **kwargs: Additional arguments.
    """

    def __init__(self, kb, **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, string: str, node_ids: List[int]) -> List[float]:
        """
        Compute exact match scores between the input string and the full information of nodes in the knowledge base.

        Args:
            string (str): The input string to compare.
            node_ids (List[int]): List of node IDs in the knowledge base.

        Returns:
            List[float]: List of exact match scores for each node.
        """
        docs = [self.kb.get_doc_info(node_id, add_rel=False, compact=True) for node_id in node_ids]
        return [int(string.lower() in doc.lower()) for doc in docs]

    def __str__(self):
        return 'compute_exact_match_score(string: str, node_ids: List[int]) -> exact_match_score: List[float].'

    def __repr__(self):
        return ("For each node in `node_ids`, compute the exact match score based on whether `string` is included in the information of the node. "
                "For example, compute_exact_match_score(string='H&M', node_ids=[2000, 3000]) returns a list of exact match scores, e.g., [0, 1], "
                "indicating that 'H&M' is included in the full information of the brand node with ID 3000 but not in the brand node with ID 2000.")


class ComputeF1Direct(Tool): 
    """
    Class to compute F1 score for a given string against a list of other strings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @format_checked
    def __call__(self, string_to_match: str, strings: List[str]) -> List[float]:
        """
        Compute F1 scores between the input string and a list of other strings.

        Args:
            string_to_match (str): The input string to compare.
            strings (List[str]): List of strings to compare against.

        Returns:
            List[float]: List of F1 scores for each string.
        """
        return [f1_score(string_to_match.lower(), s.lower()) for s in strings]

    def __str__(self):
        return 'compute_f1(string_to_match: str, strings: List[str]) -> f1_match_score: List[float]'

    def __repr__(self):
        return ("Compute the F1 score based on the similarity between `string_to_match` and each string in `strings`. "
                "For example, compute_f1(string_to_match='Adidas', strings=['Adidas', 'Adidas Originals']) returns [1, 0.67], "
                "indicating that 'Adidas' is fully matched with 'Adidas' and partially matched with 'Adidas Originals'.")


class ComputeRecallDirect(Tool): 
    """
    Class to compute recall score for a given string against a list of other strings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @format_checked
    def __call__(self, string_to_match: str, strings: List[str]) -> List[float]:
        """
        Compute recall scores between the input string and a list of other strings.

        Args:
            string_to_match (str): The input string to compare.
            strings (List[str]): List of strings to compare against.

        Returns:
            List[float]: List of recall scores for each string.
        """
        return [recall_score(string_to_match.lower(), s.lower()) for s in strings]

    def __str__(self):
        return 'compute_recall(string_to_match: str, strings: List[str]) -> recall: List[float]'

    def __repr__(self):
        return ("Compute the recall score based on the similarity between `string_to_match` and each string in `strings`. "
                "For example, compute_recall(string_to_match='H&M', strings=['Adidas', 'H&M brand']) returns [0, 1], "
                "indicating that 'H&M' is not matched with 'Adidas' but is fully included in 'H&M brand'.")


class ComputeExactMatchDirect(Tool): 
    """
    Class to compute exact match score for a given string against a list of other strings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @format_checked
    def __call__(self, string_to_match: str, strings: List[str]) -> List[float]:
        """
        Compute exact match scores between the input string and a list of other strings.

        Args:
            string_to_match (str): The input string to compare.
            strings (List[str]): List of strings to compare against.

        Returns:
            List[float]: List of exact match scores for each string.
        """
        return [int(string_to_match.lower() == s.lower()) for s in strings]

    def __str__(self):
        return 'compute_exact_match(string_to_match: str, strings: List[str]) -> exact_match_score: List[float].'

    def __repr__(self):
        return ("Compute the exact match score based on whether `string_to_match` is included in each string in `strings`. "
                "For example, compute_exact_match(string_to_match='H&M', strings=['Adidas', 'H&M']) returns [0, 1], "
                "indicating that 'H&M' is different from 'Adidas' but is the same as 'H&M'.")
