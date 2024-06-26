from typing import List, Dict
from avatar.utils.format import format_checked
from avatar.tools.tool import Tool


class GetRelatedNodes(Tool):
    """
    A class to get related nodes based on a given relation type.

    Args:
        kb: The knowledge base containing the node and relation information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'rel_type_lst'), "kb must have a method 'rel_type_lst'"
        assert hasattr(kb, 'get_neighbor_nodes'), "kb must have a method 'get_neighbor_nodes'"
        self.relation_list = self.kb.rel_type_lst()

    @format_checked
    def __call__(self, node_id: int, relation_type: str) -> List[int]:
        """
        Get related nodes based on a given relation type.

        Args:
            node_id (int): The ID of the node.
            relation_type (str): The type of relation to consider.

        Returns:
            List[int]: A list of IDs of related nodes.
        """
        assert relation_type in self.relation_list, f'relation_type must be in {self.relation_list}, but got {relation_type}'
        nodes = self.kb.get_neighbor_nodes(node_id, relation_type)
        print('get_related_nodes len', len(nodes))
        return nodes

    def __str__(self):
        return 'get_related_nodes(node_id: int, relation_type: str) -> node_ids: List[int]'

    def __repr__(self):
        return (
            "This function extracts IDs of nodes related to `node_id` based on the given `relation_type`. The returned result is a list of node IDs. "
            "For example, get_related_nodes(node_id=2000, relation_type='has_brand') returns a list of node IDs, e.g., [10, 20, 25], that are products under the brand."
        )


class GetRelationTypes(Tool):
    """
    A class to get relation types of a node.

    Args:
        kb: The knowledge base containing the node and relation information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'rel_type_lst'), "kb must have a method 'rel_type_lst'"
        assert hasattr(kb, 'get_neighbor_nodes'), "kb must have a method 'get_neighbor_nodes'"

    @format_checked
    def __call__(self, node_id: int) -> List[str]:
        """
        Get relation types of a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            List[str]: A list of relation types.
        """
        rel_types = []
        for rel_type in self.kb.rel_type_lst():
            nodes = self.kb.get_neighbor_nodes(node_id, rel_type)
            if len(nodes) > 0:
                rel_types.append(rel_type)
        print('get_relation_types', rel_types)
        return rel_types

    def __str__(self):
        return 'get_relation_types(node_id: int) -> relation_types: List[str]'

    def __repr__(self):
        return (
            "This function extracts relation types of the node with ID `node_id`. The returned result is a list of relation types. "
            "For example, get_relation_types(node_id=2000) returns a list of relation types, e.g., ['has_brand', 'also_view', 'also_buy'], "
            "of the node with ID 2000. get_relation_types(node_id=3000) may return ['also_view'] if the node with ID 3000 only has the 'also_view' relation type. "
            "If the node does not have any relation types, then it returns an empty list."
        )


class GetRelationDict(Tool):
    """
    A class to get relation types and related nodes of a node.

    Args:
        kb: The knowledge base containing the node and relation information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'rel_type_lst'), "kb must have a method 'rel_type_lst'"
        assert hasattr(kb, 'get_neighbor_nodes'), "kb must have a method 'get_neighbor_nodes'"

    @format_checked
    def __call__(self, node_id: int) -> Dict[str, List[int]]:
        """
        Get relation types and related nodes of a node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            Dict[str, List[int]]: A dictionary with relation types as keys and lists of related nodes as values.
        """
        relation_dict = {}
        for rel_type in self.kb.rel_type_lst():
            nodes = self.kb.get_neighbor_nodes(node_id, rel_type)
            relation_dict[rel_type] = nodes
        return relation_dict

    def __str__(self):
        return 'get_relation_dict(node_id: int) -> relation_dict: Dict[str, List[int]]'

    def __repr__(self):
        return (
            "This function extracts relation types and related nodes of the node with ID `node_id`. The returned result is a dictionary with relation types as keys "
            "and lists of related nodes as values. For example, get_relation_dict(node_id=2000) returns a dictionary, e.g., {'has_brand': [10], 'also_view': [2001, 2002], 'also_buy': [2003]}, "
            "for the node with ID 2000. get_relation_dict(node_id=3000) may return {'has_brand': [], 'also_view': [303], 'also_buy': []} if the node with "
            "ID 3000 only has an 'also_view' relation with the node with ID 303."
        )
