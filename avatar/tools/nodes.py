from typing import List
from avatar.tools.tool import Tool
from avatar.utils.format import format_checked


class GetNodeIDs(Tool):
    """
    A class to retrieve all node IDs of a specified type from the knowledge base.

    Args:
        kb: The knowledge base object containing the node information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'node_type_lst'), "kb must have a method 'node_type_lst'"
        assert hasattr(kb, 'get_node_ids_by_type'), "kb must have a method 'get_node_ids_by_type'"

    @format_checked
    def __call__(self, node_type: str) -> List[int]:
        """
        Retrieves all node IDs of the specified type.

        Args:
            node_type (str): The type of nodes to retrieve.

        Returns:
            List[int]: A list of node IDs of the specified type.
        """
        assert node_type in self.kb.node_type_lst(), f'node_type must be in {self.kb.node_type_lst()}, but got {node_type}'
        return self.kb.get_node_ids_by_type(node_type)

    def __str__(self):
        return 'get_node_ids_by_type(node_type: str) -> node_ids: List[int]'

    def __repr__(self):
        return 'Return a list containing all of the IDs of nodes with type `node_type`.'


class GetNodeType(Tool):
    """
    A class to retrieve the type of a specified node ID from the knowledge base.

    Args:
        kb: The knowledge base object containing the node information.
    """

    def __init__(self, kb, **kwargs):
        super().__init__(kb=kb)
        assert hasattr(kb, 'get_node_type_by_id'), "kb must have a method 'get_node_type_by_id'"

    @format_checked
    def __call__(self, node_id: int) -> str:
        """
        Retrieves the type of the specified node ID.

        Args:
            node_id (int): The ID of the node to retrieve the type for.

        Returns:
            str: The type of the specified node.
        """
        return self.kb.get_node_type_by_id(node_id)

    def __str__(self):
        return 'get_node_type_by_id(node_id: int) -> node_type: str'

    def __repr__(self):
        return 'Return a string representing the node type of the node with id `node_id`.'
