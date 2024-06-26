import os
import os.path as osp
import json
from typing import List, Dict
from avatar.utils.format import format_checked
from stark_qa.tools.api import get_llm_output
from avatar.tools.tool import Tool


class QueryParser(Tool):
    """
    A class to parse a query into a dictionary of attributes using a specified parser model.

    Args:
        parser_model (str): The model to use for parsing the query.
    """

    def __init__(self, parser_model: str, **kwargs):
        self.parser_model = parser_model
        super().__init__()

    @format_checked
    def __call__(self, query: str, attributes: List[str]) -> Dict[str, str]:
        """
        Parses the query to extract the specified attributes.

        Args:
            query (str): The query string to be parsed.
            attributes (List[str]): A list of attribute names to extract from the query.

        Returns:
            Dict[str, str]: A dictionary where keys are attribute names and values are the extracted attributes.
        """
        prompt = (
            "You are a helpful assistant that helps me extract attributes from a given query. "
            "This is the query: \"<query>\"\nThese are the attribute names: \n<attributes>\n"
            "Please output a JSON dictionary where the keys are the attribute names (string) and each value is the corresponding "
            "extracted attribute (string) from the query. If an attribute is not mentioned in the query, the value should be \"NA\". "
            "Your output: "
        )
        prompt = prompt.replace('<query>', query)
        prompt = prompt.replace('<attributes>', str(attributes))
        
        while True:
            output = get_llm_output(prompt, model=self.parser_model, json_object=True)
            output = json.loads(output)
            try:
                assert set(list(output.keys())) == set(attributes)
                break
            except Exception as e:
                print(f'parse_query - keys do not match attributes: {output.keys()} != {attributes}')
                pass
        
        print('parse_query - query', query, 'attributes', attributes)
        print('parse_query - output', output)
        return output

    def __str__(self):
        return 'parse_query(query: str, attributes: List[str]) -> Dict[str, str]'

    def __repr__(self):
        return (
            "This function parses a `query` into a dictionary based on the input list `attributes`. In the output dictionary, each key is an attribute name from `attributes`, "
            "and the value is a string corresponding to the extracted attribute from the query. If an attribute is not mentioned in the query, the value should be 'NA'. "
            "For example, for a query 'Can you recommend me a durable soccer rebounder for beginner that is from VEVRO?' and an attribute list "
            "['product_type', 'brand', 'user_level', 'property', 'price'], the output dictionary could be "
            "{'product_type': 'soccer rebounder', 'brand': 'VEVRO', 'user_level': 'beginner', 'property': 'durable', 'price': 'NA'}."
        )
