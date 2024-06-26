import re
import sys
import os
import torch
import numpy as np
import os.path as osp
from typing import List, Dict, Union
import PIL
import json
from tqdm import tqdm

from avatar.utils.format import format_checked
from avatar.tools.text_extraction import GetRelevantChunk
from avatar.tools.tool import Tool
from stark_qa.tools.api import get_llm_output, get_llm_outputs
from avatar.utils.api_vision import get_llm_vision_outputs

MAX_RETRY = 5

class LLMSummarize(Tool):
    """
    A class to use LLM to summarize each text in `texts` into a summary with no more than `max_length` words.
    
    Args:
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
    """

    def __init__(self, model_name: str, n_limit: int = 100, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        super().__init__()

    @format_checked
    def __call__(self, texts: Union[str, List[str]], max_length: int = 50) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        print(f'summarize_texts_by_llm - len(texts)={len(texts)}')
        prompts = [
            (f'You are a helpful assistant that summarizes the following text into no more than {max_length} words. '
             f'Here is the text:\n"{text}"\n'
             f'Please provide a summary that is no longer than {max_length} words. You should output the summary only without additional comments. The summary is:')
            for text in texts
        ]

        summaries = get_llm_outputs(prompts, model=self.model_name, temperature=0.2)
        summaries = [summary.strip(' "\'\n') for summary in summaries]
        return summaries

    def __str__(self):
        return 'summarize_texts_by_llm(texts: Union[str, List[str]], max_length: int=50) -> summaries: List[str]'

    def __repr__(self):
        return ("Use LLM to summarize each text in `texts` into a summary with no more than `max_length` words. "
                "The returned result is a list. For example, summarize_texts_by_llm(texts=['<long text about a bike product>', '<long text about soccer>'], "
                "max_length=50) returns a list ['<A short summary about a bike product>', '<A short summary about soccer>']. "
                f"This operation can be used for at most {self.n_limit} times. For efficiency, use this function with multiple texts at once, and avoid calling it separately for each text.")


class LLMExtractInfo(Tool):
    """
    A class to use LLM to extract information relevant to `extract_term` from each text in the given `texts`.

    Args:
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
    """

    def __init__(self, model_name: str, n_limit: int = 100, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        super().__init__()

    @format_checked
    def __call__(self, texts: Union[str, List[str]], extract_term: str) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        print(f'extract_relevant_info_by_llm - len(texts)={len(texts)} - extract_term={extract_term}')
        prompts = [
            (f'You are a helpful assistant that extracts information relevant to "{extract_term}" from the following information. '
             f'Here is the information:\n{text}\n\n'
             f'Assume you have no other knowledge and should only use the text provided. Please extract the information relevant to "{extract_term}" from the above information. If there is no information relevant to "{extract_term}", please output "NA". Output your answer without additional comments: ')
            for text in texts
        ]

        extracted_info = get_llm_outputs(prompts, model=self.model_name, temperature=0.2)
        extracted_info = [info.strip(' "\'\n') for info in extracted_info]
        return extracted_info

    def __str__(self):
        return 'extract_relevant_info_by_llm(texts: Union[str, List[str]], extract_term: str) -> extracted_info: List[str]'

    def __repr__(self):
        return (f"Use LLM to extract information relevant to `extract_term` from each text in the given `texts`. "
                "The returned result is a list of strings. For example, extract_relevant_info_by_llm(texts=['product name: Soccer Rebounder, ...review: My kids loved it and I didn\'t need to look after them since it was completely safe...'], "
                "extract_term='safe for kids') returns a list containing the string ['My kids loved it and I didn\'t need to look after them since it was completely safe'] from the information in `texts`. "
                f"If there is no relevant information, the function returns 'NA'. This operation can be used for at most {self.n_limit} times. "
                "For efficiency, use this function with multiple texts at once and avoid calling it separately for each text.")


class LLMCheck(Tool):
    """
    A class to use LLM to check if node(s) with `node_ids` satisfy the `requirement`.

    Args:
        kb: The knowledge base containing node information.
        model_name (str): The name of the LLM model to use.
        chunk_size (int): The size of chunks for processing.
        chunk_emb_dir (str): The directory to save or load chunk embeddings.
        n_limit (int): The maximum number of times this function can be used.
        initial_temperature (float): The initial temperature for the LLM model.
        use_chunk (bool): Whether to use chunking for processing.
    """

    def __init__(self, kb, 
                 model_name: str, chunk_size: int = None, chunk_emb_dir: str = None,
                 n_limit: int = 100, initial_temperature: float = 0.2, use_chunk: bool = False, **kwargs):
        assert hasattr(kb, 'get_doc_info'), "kb must have a method 'get_doc_info'"
        assert hasattr(kb, 'get_node_type_by_id'), "kb must have a method 'get_node_type_by_id'"
        self.n_limit = n_limit
        self.use_chunk = use_chunk
        self.model_name = model_name
        self.initial_temperature = initial_temperature
        if use_chunk:
            self.chunk_emb_dir = chunk_emb_dir
            self.chunk_size = chunk_size
            self.chunk_tool = GetRelevantChunk(kb, chunk_emb_dir=chunk_emb_dir)
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, node_ids: Union[int, List[int]], requirement: str, return_rationale: bool = False) -> List[bool]:
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        prompts = {}
        for node_id in node_ids:
            if self.use_chunk:
                doc = self.chunk_tool(node_id, requirement, k=10, chunk_size=self.chunk_size)
            else:
                doc = self.kb.get_doc_info(node_id, add_rel=True, compact=False)
            node_type = self.kb.get_node_type_by_id(node_id)
            prompt = (
                f'Your task is to check if a {node_type} meets the following requirement:\n'
                f' "{requirement}"\n\n'
                f'I will provide you with the information of the {node_type}:\n{doc}\n\n'
                f'Respond according to these guidelines:\n\n'
                f'1. If the requirement is met, present evidence and end with "=> yes". Format: "evidence => yes". '
                f'Example: For a requirement "Is the product from Radio Flyer safe for kids?" with information "Brand: Radio Flyer; Safety certifications: EN71", '
                f'respond "the product is from Radio Flyer and the safety certifications indicate it\'s safe for kids => yes".\n\n'
                f'2. If the requirement is not met, explain the gap and end with "=> no". Format: "your reason => no". '
                f'Example: If the requirement includes "safe for kids and includes installation tools" but there\'s no mention of tools, '
                f'respond "it doesn\'t mention the installation tools => no".\n\n'
                f'Note that you should break down the requirement if multiple demands are presented and find the evidence individually. Please output your answer in the format described above. Use "=>" only once to indiate your final decision and avoid adding any additional comments after "yes" or "no". Does this {node_type} meet the requirement "{requirement}"?\nYour output:'
            )
            prompts[node_id] = prompt

        cnt = 0
        temperature = self.initial_temperature
        scores, rationale = {}, {}
        print(f'check_req_by_llm - {len(node_ids)} nodes to check')
        while len(scores) < len(node_ids):
            cnt += 1
            if cnt > MAX_RETRY // 5:
                temperature = min(2 * temperature, 1)
            if cnt > MAX_RETRY // 2:
                temperature = min(2 * temperature, 1)

            todo_node_ids = [node_id for node_id in node_ids if node_id not in scores.keys()]
            prompt_lst = [prompts[node_id] for node_id in todo_node_ids]
            answers = get_llm_outputs(prompt_lst, model=self.model_name, temperature=temperature)

            for node_id, answer in zip(todo_node_ids, answers):
                if '=>' in answer:
                    parsed_answer = answer.split('=>')[1].strip(' "\'\n').split(' ')[0].strip('. "\'\n')
                    if len(parsed_answer) < len('yes') + 1 and len(parsed_answer) > 1:
                        if 'yes' in parsed_answer.lower():
                            scores[node_id] = True
                        elif 'no' in parsed_answer.lower():
                            scores[node_id] = False
                        else:
                            if cnt > MAX_RETRY:
                                scores[node_id] = False
                            continue
                        rationale[node_id] = answer.split('=>')[0].strip(' \n')
                        print('check_req_by_llm - node_id', node_id, 'answer', parsed_answer)
                else:
                    print('check_req_by_llm - failed | ', answer)

        score_lst = [scores[node_id] for node_id in node_ids]
        rationale_lst = [rationale[node_id] for node_id in node_ids]
        if return_rationale:
            return score_lst, rationale_lst
        return score_lst

    def __str__(self):
        return 'check_req_by_llm(node_ids: Union[int, List[int]], requirement: str) -> result: List[bool]'

    def __repr__(self):
        return (f"Use LLM to check if node(s) with `node_ids` satisfy the `requirement`. The returned result is a list of boolean values. "
                "For example, check_req_by_llm(node_ids=[1024], requirement='safe for kids') returns [True] if the LLM confirms that the product node with node ID 1024 is safe for kids, or [False] otherwise. "
                "For efficiency, use this function with multiple node IDs at once and avoid calling it separately for each node ID.")


class LLMClassification(Tool):
    """
    A class to use LLM to classify the given `text` into one of the given `classes`.

    Args:
        kb: The knowledge base containing node information.
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
        initial_temperature (float): The initial temperature for the LLM model.
    """

    def __init__(self, kb, model_name: str, n_limit: int = 100, initial_temperature: float = 0.2, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        self.initial_temperature = initial_temperature
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, text: str, classes: List[str]) -> str:
        assert len(classes) > 1, f'classes must have at least 2 elements, but got {len(classes)}'

        cnt = 0
        temperature = self.initial_temperature
        print(f'classify_by_llm - classes', classes)
        while True:
            cnt += 1
            if cnt > MAX_RETRY // 5:
                temperature = min(2 * temperature, 1)
            if cnt > MAX_RETRY // 2:
                temperature = min(2 * temperature, 1)

            str_classes = '\n'.join([f'{i + 1}: {cls}' for i, cls in enumerate(classes)])
            prompt = (
                f'You are a helpful assistant that classifies a text into the most appropriate class in a list of given classes. \n'
                f'This is the text information:\n\n"{text}"\n\n'
                f'Based on the text content, please classify the text into one of the following classes: \n{str_classes}\n'
                f'Your output should be the class index (a number) only without additional comments. The class index: '
            )

            pred_class = get_llm_output(prompt, self.model_name, temperature=temperature)
            try:
                pred_class = int(eval(pred_class) - 1)
                if pred_class >= 0 and pred_class < len(classes):
                    break
            except:
                pass
            if cnt > MAX_RETRY:
                return 'NA'
        print(f'classify_by_llm - pred_class', pred_class)
        return classes[pred_class]

    def __str__(self):
        return 'classify_by_llm(text: str, classes: List[str]) -> pred_class: str'

    def __repr__(self):
        return (f"Use LLM to classify the given `text` into one of the given `classes`. The returned result is a string. "
                "For example, classify_by_llm(text='I have to keep it away from my kids in the kitchen', classes=['safe', 'unsafe']) returns 'unsafe'. "
                "classify_by_llm(text='diabetes', classes=['disease', 'drug']) returns 'disease'. "
                "classify_by_llm(text='<a query>', classes=['This query only involves single-hop relation between ...', 'This query involves multi-hop reasoning between ...', 'none of the above']) may return 'none of the above' if `text` is not a single-hop or multi-hop query. "
                f"If the LLM cannot classify the text into any of the given classes, the function returns 'NA'. This operation can be used for at most {self.n_limit} times.")


class LLMClassifyNode(Tool):
    """
    A class to use LLM to classify each node specified by `node_ids` into one of the given `classes` or 'NA'.

    Args:
        kb: The knowledge base containing node information.
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
        initial_temperature (float): The initial temperature for the LLM model.
    """

    def __init__(self, kb, model_name: str, n_limit: int = 100, initial_temperature: float = 0.2, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        self.initial_temperature = initial_temperature
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, node_ids: Union[int, List[int]], classes: List[str]) -> List[str]:
        assert len(classes) > 1, f'classes must have at least 2 elements, but got {len(classes)}'
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        cnt = 0
        pred_classes = {}
        temperature = self.initial_temperature
        while len(pred_classes) < len(node_ids):
            cnt += 1
            if cnt > MAX_RETRY // 5:
                temperature = min(2 * temperature, 1)
            if cnt > MAX_RETRY // 2:
                temperature = min(2 * temperature, 1)

            str_classes = '\n'.join([f'{i + 1}: {cls}' for i, cls in enumerate(classes)])
            prompts = [
                (f'You are a helpful assistant that classifies a text into the most appropriate class in a list of given classes. \n'
                 f'This is the text information:\n\n"{self.kb.get_doc_info(node_id, add_rel=False, compact=False)}"\n\n'
                 f'Based on the text content, please classify the text into one of the following classes: \n{str_classes}\n'
                 f'Your output should be the class index (a number) only without additional comments. The class index: ')
                for node_id in node_ids
            ]

            pred_class = get_llm_outputs(prompts, model=self.model_name, temperature=temperature)
            try:
                for node_id, pred in zip(node_ids, pred_class):
                    pred = int(eval(pred) - 1)
                    if pred >= 0 and pred < len(classes):
                        pred_classes[node_id] = classes[pred]
            except:
                pass
            if cnt > MAX_RETRY:
                break
        for node_id in node_ids:
            if node_id not in pred_classes.keys():
                pred_classes[node_id] = 'NA'
        return [pred_classes[node_id] for node_id in node_ids]

    def __str__(self):
        return 'classify_nodes_by_llm(node_ids: Union[int, List[int]], classes: List[str]) -> pred_classes: List[str]'

    def __repr__(self):
        return (f"Use LLM to classify each node specified by `node_ids` into one of the given `classes` or 'NA'. The returned result is a list. "
                "For example, classify_nodes_by_llm(node_ids=[10024, 10025], classes=['hat', 'shirt']) may returns ['shirt', 'NA'] if the LLM classifies the first node as 'shirt' and cannot classify the second node. "
                "classify_nodes_by_llm(node_ids=[20001], classes=['disease', 'drug']) may returns []'disease']. "
                f"If the LLM cannot classify the text into any of the given classes, the function returns 'NA'. This operation can be used for at most {self.n_limit} times.")


class LLMScore(Tool):
    """
    A class to use LLM to score the node(s) with `node_ids` based on the given `query`.

    Args:
        kb: The knowledge base containing node information.
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
        initial_temperature (float): The initial temperature for the LLM model.
    """

    def __init__(self, kb, model_name: str, n_limit: int = 100, initial_temperature: float = 0.2, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        self.initial_temperature = initial_temperature
        super().__init__(kb=kb)

    @format_checked
    def __call__(self, node_ids: Union[int, List[int]], query: str) -> List[float]:
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        prompts = {}
        for node_id in node_ids:
            node_type = self.kb.get_node_type_by_id(node_id)
            prompt = (
                f'You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0 based on the degree of satisfaction. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n'
                f'Here is the query:\n"{query}"\n'
                f'Here is the information about the {node_type}:\n' +
                self.kb.get_doc_info(node_id, add_rel=True) + '\n\n' +
                f'Please score the {node_type} based on how well it satisfies the query. Your output format should be "your reasoning process => score". '
                f'For example, the output could be "The product is safe for kids based on the safety certifications and the reviews => 1.0\", or "The product is safe but there is no evidence on its installation tools => 0.5". '
                f'Please output your answer in the format described above. Use "=>" only once to indiate your final score and avoid adding any additional comments after the score. Your output: '
            )
            prompts[node_id] = prompt

        scores = {}
        cnt = 0
        temperature = self.initial_temperature
        while len(scores) < len(node_ids):
            cnt += 1
            if cnt > MAX_RETRY // 5:
                temperature = min(2 * temperature, 1)
            if cnt > MAX_RETRY // 2:
                temperature = min(2 * temperature, 1)

            todo_node_ids = [node_id for node_id in node_ids if node_id not in scores.keys()]
            prompt_lst = [prompts[node_id] for node_id in todo_node_ids]
            answers = get_llm_outputs(prompt_lst, model=self.model_name, temperature=temperature)

            for node_id, answer in zip(todo_node_ids, answers):
                if '=>' in answer:
                    score = answer.split('=>')[1].strip(' "\'\n')
                    if len(score) > 1:
                        try:
                            scores[node_id] = find_floating_number(score)[0]
                        except:
                            if cnt > MAX_RETRY:
                                scores[node_id] = 0.0
        score_lst = [scores[node_id] for node_id in node_ids]
        return score_lst

    def __str__(self):
        return 'get_scores_by_llm(node_ids: Union[int, List[int]], query: str) -> scores: List[float]'

    def __repr__(self):
        return (f"Use LLM to score the node(s) with `node_ids` based on the given `query`. The returned result is a list of float numbers between 0.0 and 1.0 (inclusive). "
                "For example, get_scores_by_llm(node_ids=[1024, 1025], query='Is the product safe for kids?') may return [1.0, 0.3]. "
                "Specifically, a value in the output list is 1.0 if the LLM is absolutely sure that the corresponding product node is safe for kids, "
                "0.0 if the LLM confirms that the product node is not safe for kids or there is no evidence showing the product is safe for kids, "
                f"or a float number between 0.0 and 1.0 if partial or weak evidence exists. This operation can be used for at most {self.n_limit} times. "
                "For efficiency, use this function with multiple node IDs at once and avoid calling it separately for each node ID.")


class LLMVQA(Tool):
    """
    A class to use LLM to answer the given `question` based on the image(s) in the given `image_lst`.

    Args:
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
    """

    def __init__(self, model_name: str, n_limit: int = 10, **kwargs):
        self.n_limit = n_limit
        self.model_name = model_name
        super().__init__()

    @format_checked
    def __call__(self, question: str, image_lst: List[PIL.Image.Image]) -> List[str]:
        responses = get_llm_vision_outputs(image_lst, 
                                           message=question, 
                                           model=self.model_name, 
                                           json_object=False
                                           )
        return responses

    def __str__(self):
        return 'vqa_by_llm(question: str, image_lst: List[PIL.Image.Image]) -> responses: List[str]'

    def __repr__(self):
        return ("Use LLM to answer the given `question` based on the image(s) in the given `image_lst`. "
                "The returned result is a list of strings. For example, vqa_by_llm(question='What is the color of the car?', "
                "image_lst=[<PIL.Image.Image of a car>, <PIL.Image.Image of a truck>]) may return ['The color of the car is red', 'The truck is blue']. "
                f"This operation can be used for at most {self.n_limit} times. For efficiency, use this function with multiple images at once and avoid calling it separately for each image.")


class LLMVisualAttribute(Tool):
    """
    A class to use LLM to extract attributes about the given `attribute_lst` from each image in the given `image_lst`.

    Args:
        model_name (str): The name of the LLM model to use.
        n_limit (int): The maximum number of times this function can be used.
    """

    def __init__(self, model_name: str, n_limit: int = 10, **kwargs):
        self.model_name = model_name
        self.n_limit = n_limit
        super().__init__()

    @format_checked
    def __call__(self, attribute_lst: List[str], image_lst: List[PIL.Image.Image]) -> List[Dict[str, str]]:
        prompt = (f'You are a helpful assistant that extracts attributes about {attribute_lst} from an image. '
                  f'Your output should be a JSON dictionary where the keys are the attribute names (string) and each value is the corresponding extracted attribute (string) from the image. '
                  f'If an attribute is not mentioned in the image, the value should be "NA". Please make sure the keys exactly contain and match the attribute names in the list.')
        responses = get_llm_vision_outputs(image_lst, 
                                           message=prompt, 
                                           model=self.model_name, 
                                           json_object=True
                                           )
        responses = [json.loads(response) for response in responses]
        return responses

    def __str__(self):
        return 'extract_visual_attributes_by_llm(attribute_lst: List[str], image_lst: List[PIL.Image.Image]) -> responses: List[Dict[str, str]]'

    def __repr__(self):
        return ("Use LLM to extract attributes about the given `attribute_lst` from each image in the given `image_lst`. "
                "The returned result is a list of dictionaries. For example, extract_visual_attributes_by_llm(attribute_lst=['vehicle color', 'number of people'], "
                "image_lst=[<PIL.Image.Image of a car>, <PIL.Image.Image of a cat>]) may return [{'vehicle color': 'red', 'number of people': '2'}, {'vehicle color': 'NA', 'number of people': '1'}]. "
                f"where 'NA' indicates that an attribute is not mentioned in the image. This operation can be used for at most {self.n_limit} times. "
                "For efficiency, use this function with multiple images and multiple attributes at once and avoid calling it separately for each image and attribute.")


def find_floating_number(text: str) -> List[float]:
    """
    Find all floating point numbers in the given text.

    Args:
        text (str): The text to search for floating point numbers.

    Returns:
        List[float]: A list of floating point numbers found in the text.
    """
    pattern = r'0\.\d+|1\.0'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]

