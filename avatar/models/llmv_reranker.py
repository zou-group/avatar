import torch
from typing import Any, List
import re

from avatar.models.vss import VSS
from avatar.models.model import ModelForQA
from avatar.utils.api_vision import get_llm_vision_outputs
import re


def find_floating_number(text: str) -> List[float]:
    """
    Extract floating point numbers from the given text.

    Args:
        text (str): Input text from which to extract numbers.

    Returns:
        List[float]: List of extracted floating point numbers.
    """
    pattern = r'0\.\d+|1\.0'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]


class LLMvReranker(ModelForQA):
    
    def __init__(self,
                 kb, 
                 model_name: str,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 sim_weight: float = 1,
                 max_k: int = 100
                 ):
        '''
        Answer the query by VLM model.
        Args:
            kb (Any): kb
            model_name (str): model name
            query_emb_dir (str): query embedding directory
            candidates_emb_dir (str): candidates embedding directory
            sim_weight (float): similarity weight
            max_k (int): maximum number of top candidates to consider  
        '''
        
        super().__init__(kb)
        self.max_k = max_k
        self.model_name = model_name
        self.sim_weight = sim_weight

        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir)

    def forward(self, 
                query,
                query_id=None,
                **kwargs: Any):
        
        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())
        # get the ids with top k highest scores
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        cand_len = len(top_k_node_ids)

        prompt = (
            f'You are a helpful assistant that examines if an image satisfies a given query and assign a score from 0.0 to 1.0 based on the degree of satisfaction. If the image does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that image satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n'
            f'Here is the query:\n\"{query}\"\n'
            f'Please score the image based on how well it satisfies the query. Only output the floating point score without anything else. '
            f'The numeric score of this image is: '
            )
        pred_dict = {}
        images = []
        for idx, node_id in enumerate(top_k_node_ids):
            images.append(self.kb.get_image(node_id))

        answers = get_llm_vision_outputs(image_list=images, message=prompt, model=self.model_name)
        for idx, (node_id, str_answer) in enumerate(zip(top_k_node_ids, answers)):
            answer = find_floating_number(str_answer)
            if len(answer) == 1:
                answer = answer[0]
            else:
                answer = 0.0
                print('answer length not 1, redoing...')
            gpt_score = float(answer)
            sim_score = (cand_len - idx) / cand_len
            score = gpt_score + self.sim_weight * sim_score
            pred_dict[node_id] = score
            print('llm_score', gpt_score, 'sim_score', sim_score)

        print('pred_dict', pred_dict)
        return pred_dict
        
    