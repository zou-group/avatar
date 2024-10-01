import torch
import traceback
import json
import os
import re
import os.path as osp
import requests
from PIL import Image
from typing import Any, Dict

from avatar.tools import assigned_funcs, customized_funcs, general_funcs
from avatar.utils.timer import exit_after

from avatar.models.vss import VSS
from avatar.models.model import ModelForQA
from stark_qa.tools.api import get_openai_embedding
from avatar.tools.react.api import get_llm_output_tools
from avatar.utils.error_handler import string_exec_error_handler
from avatar.utils.image import image_to_base64

class ReactEnv:
    def __init__(
        self,
        database,
        chunk_size,
        chunk_emb_dir,
        node_emb_dir,
        model_name,
        emb_model,
        debug_print_path,
        dataset,
        n_limit=100,
        initial_temperature=0.2,
        n_init_candidates=20,
        use_chunk=False,
    ):
        """
        Initialize the environment.
        """
        super().__init__()
        self.page = None  # current Wikipedia page
        self.obs = None  # current observation
        self.obs_img = None  # current image observation
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.search_time = 0
        self.num_searches = 0
        self.database = database
        self.chunk_size = chunk_size
        self.chunk_emb_dir = chunk_emb_dir
        self.node_emb_dir = node_emb_dir
        self.api_func_llm = model_name
        self.emb_model = emb_model
        self.debug_print_path = debug_print_path
        self.n_limit = n_limit
        self.initial_temperature = initial_temperature
        self.use_chunk = use_chunk
        self.dataset = dataset
        self.n_init_candidates = n_init_candidates
        self.APIs = self._get_APIs()
        self.embedding_list = []


    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def _get_APIs(self) -> Dict[str, Any]:
        assigned_funcs_key = self.dataset if self.dataset not in ['amazon', 'prime', 'mag'] else 'stark'
        apis = assigned_funcs[assigned_funcs_key]
        available_funcs = general_funcs
        if self.dataset in customized_funcs.keys():
            available_funcs.update(customized_funcs[self.dataset])

        kwargs_union = {
            'kb': self.database,
            'emb_model': self.emb_model,
            'model_name': self.api_func_llm,
            'parser_model': self.api_func_llm, 
            'chunk_size': self.chunk_size,
            'chunk_emb_dir': self.chunk_emb_dir,
            'node_emb_dir': self.node_emb_dir,
            'debug_print_path': self.debug_print_path,
            'n_limit': self.n_limit
        }
        variables = {}
        for api in apis:
            assert api in available_funcs, f'API {api} is not available for this dataset!' \
                   ' Please either remove it from assignment or add it to registered/customized functions!'
            func = available_funcs[api](**kwargs_union)
            variables[api] = func
            if api == 'debug_print':
                func.clean_file()

        self.funcs = list(variables.values())
        variables.update({'exit_after': exit_after})
        return variables

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.obs = (
            "Interact with knowledge base with given API.\n"
        )
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action_name, action_params):
        print(f'API calling {action_name=}')
        print(f'API parameters {action_params=}')
        reward = 0
        done = False
        # action = action.strip()
        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()
        obs_img = None
        # possible action_names are self.APIs.keys()
        if action_name == "parse_query":
            query = action_params["query"]
            attributes = action_params["attributes"]
            self.obs = self.APIs["parse_query"](query=query, attributes=attributes)
        elif action_name == "get_node_ids_by_type":
            node_type = action_params["node_type"]
            self.obs = self.APIs["get_node_ids_by_type"](node_type=node_type)
        elif action_name == "get_node_type_by_id":
            node_id = action_params["node_id"]
            self.obs = self.APIs["get_node_type_by_id"](node_id=node_id)
        elif action_name == "get_full_info":
            node_id = action_params["node_id"]
            self.obs = self.APIs["get_full_info"](node_id=node_id)
        elif action_name == "get_text_info":
            if self.dataset == 'flickr30k_entities':
                image_ids = action_params["image_ids"]
                self.obs = self.APIs["get_text_info"](image_ids=image_ids)
            else:
                node_id = action_params["node_id"]
                self.obs = self.APIs["get_text_info"](node_id=node_id)
        elif action_name == "get_relation_info":
            node_id = action_params["node_id"]
            self.obs = self.APIs["get_relation_info"](node_id=node_id)
        elif action_name == "get_relevant_chunk":
            node_id = action_params["node_id"]
            attribute = action_params["attribute"]
            self.obs = self.APIs["get_relevant_chunk"](
                node_id=node_id, attribute=attribute
            )
        elif action_name == "get_text_embedding":
            string = action_params["string"]
            embeddings = self.APIs["get_text_embedding"](string=string)
            n, m = embeddings.size()
            start = len(self.embedding_list)
            for i in range(n):
                self.embedding_list.append(embeddings[i])
            end = len(self.embedding_list)
            self.obs = list(range(start, end))
        elif action_name == "get_node_embedding":
            node_ids = action_params["node_ids"]
            embeddings = self.APIs["get_node_embedding"](node_ids=node_ids)
            n, m = embeddings.size()
            start = len(self.embedding_list)
            for i in range(n):
                self.embedding_list.append(embeddings[i])
            end = len(self.embedding_list)
            self.obs = list(range(start, end))
        elif action_name == "get_related_nodes":
            node_id = action_params["node_id"]
            relation_type = action_params["relation_type"]
            self.obs = self.APIs["get_related_nodes"](
                node_id=node_id, relation_type=relation_type
            )
        elif action_name == "get_relation_types":
            self.obs = self.APIs["get_relation_types"]()
        elif action_name == "get_relation_dict":
            self.obs = self.APIs["get_relation_dict"]()
        elif action_name == "compute_exact_match_score":
            string = action_params["string"]
            node_ids = action_params["node_ids"]
            self.obs = self.APIs["compute_exact_match_score"](
                string=string, node_ids=node_ids
            )
        elif action_name == "compute_recall_score":
            string = action_params["string"]
            node_ids = action_params["node_ids"]
            self.obs = self.APIs["compute_recall_score"](
                string=string, node_ids=node_ids
            )
        elif action_name == "compute_f1_score":
            string = action_params["string"]
            node_ids = action_params["node_ids"]
            self.obs = self.APIs["compute_f1_score"](string=string, node_ids=node_ids)
        elif action_name == "compute_cosine_similarity":
            embedding_1_idx = action_params["embedding_1_idx"]
            embedding_2_idx = action_params["embedding_2_idx"]
            embedding_1 = torch.stack([self.embedding_list[i] for i in embedding_1_idx])
            embedding_2 = torch.stack([self.embedding_list[i] for i in embedding_2_idx])
            self.obs = self.APIs["compute_cosine_similarity"](
                embedding_1=embedding_1, embedding_2=embedding_2
            )
        elif action_name == "compute_query_node_similarity":
            query = action_params["query"]
            node_ids = action_params["node_ids"]
            self.obs = self.APIs["compute_query_node_similarity"](
                query=query, node_ids=node_ids
            )
        elif action_name == "summarize_texts_by_llm":
            texts = action_params["texts"]
            max_length = action_params["max_length"]
            self.obs = self.APIs["summarize_texts_by_llm"](texts=texts, max_length=max_length)
        elif action_name == "classify_by_llm":
            text = action_params["text"]
            classes = action_params["classes"]
            self.obs = self.APIs["classify_by_llm"](text=text, classes=classes)
        elif action_name == "classify_nodes_by_llm":
            node_ids = action_params["node_ids"]
            classes = action_params["classes"]
            self.obs = self.APIs["classify_nodes_by_llm"](node_ids=node_ids, classes=classes)
        elif action_name == "extract_relevant_info_by_llm":
            texts = action_params["texts"]
            extract_term = action_params["extract_term"]
            self.obs = self.APIs["extract_relevant_info_by_llm"](
                texts=texts, extract_term=extract_term
            )
        elif action_name == "check_req_by_llm":
            node_ids = action_params["node_ids"]
            requirement = action_params["requirement"]
            self.obs = self.APIs["check_req_by_llm"](
                node_ids=node_ids, requirement=requirement
            )
        elif action_name == "get_scores_by_llm":
            node_ids = action_params["node_ids"]
            query = action_params["query"]
            self.obs = self.APIs["get_scores_by_llm"](node_ids=node_ids, query=query)
        elif action_name == "debug_print":
            string = action_params["string"]
            self.obs = self.APIs["debug_print"](string=string)
        elif action_name == "exit_after":
            time = action_params["time"]
            self.obs = self.APIs["exit_after"](time=time)
        elif action_name == "get_image_embedding":
            image_ids = action_params["image_ids"]
            embeddings = self.APIs["get_image_embedding"](image_ids=image_ids)
            n, m = embeddings.size()
            start = len(self.embedding_list)
            for i in range(n):
                self.embedding_list.append(embeddings[i])
            end = len(self.embedding_list)
            self.obs = list(range(start, end))
        elif action_name == "get_bag_of_phrases":
            image_ids = action_params["image_ids"]
            self.obs = self.APIs["get_bag_of_phrases"](image_ids=image_ids)
        elif action_name == "get_clip_text_embedding":
            string = action_params["string"]
            embeddings = self.APIs["get_clip_text_embedding"](string=string)
            n, m = embeddings.size()
            start = len(self.embedding_list)
            for i in range(n):
                self.embedding_list.append(embeddings[i])
            end = len(self.embedding_list)
            self.obs = list(range(start, end))
        elif action_name == "get_clip_image_embedding":
            image_lst_id = action_params["image_lst"]
            image_lst = []
            for i in image_lst_id:
                relative_image_path = f'raw/flickr30k-images/{i}.jpg'
                image = Image.open(osp.join(self.database.root, relative_image_path))
                image_lst.append(image)
            embeddings = self.APIs["get_clip_image_embedding"](image_lst=image_lst)
            n, m = embeddings.size()
            start = len(self.embedding_list)
            for i in range(n):
                self.embedding_list.append(embeddings[i])
            end = len(self.embedding_list)
            self.obs = list(range(start, end))
        elif action_name == "get_patch_id_to_phrase_dict":
            image_ids = action_params["image_ids"]
            self.obs = self.APIs["get_patch_id_to_phrase_dict"](image_ids=image_ids)
        elif action_name == "get_image_patch_by_phrase_id":
            image_id = action_params["image_id"]
            phrase_id = action_params["phrase_id"]
            self.obs = None
            obs_img = self.APIs["get_image_patch_by_phrase_id"](
                image_id=image_id, phrase_id=phrase_id
            )
            obs_img = image_to_base64(obs_img)
        elif action_name == "compute_f1":
            string_to_match = action_params["string_to_match"]
            strings = action_params["strings"]
            self.obs = self.APIs["compute_f1"](string_to_match=string_to_match, strings=strings)
        elif action_name == "compute_recall":
            string_to_match = action_params["string_to_match"]
            strings = action_params["strings"]
            self.obs = self.APIs["compute_recall"](
                string_to_match=string_to_match, strings=strings
            )
        elif action_name == "compute_exact_match":
            string_to_match = action_params["string_to_match"]
            strings = action_params["strings"]
            self.obs = self.APIs["compute_exact_match"](
                string_to_match=string_to_match, strings=strings
            )
        elif action_name == "vqa_by_llm":
            question = action_params["question"]
            image_lst_id = action_params["image_lst"]
            image_lst = []
            for i in image_lst_id:
                relative_image_path = f'raw/flickr30k-images/{i}.jpg'
                image = Image.open(osp.join(self.database.root, relative_image_path))
                image_lst.append(image)
            self.obs = self.APIs["vqa_by_llm"](question=question, image_lst=image_lst)
        elif action_name == "extract_visual_attributes_by_llm":
            attribute_lst = action_params["attribute_lst"]
            image_lst_id = action_params["image_lst"]
            image_lst = []
            for i in image_lst_id:
                relative_image_path = f'raw/flickr30k-images/{i}.jpg'
                image = Image.open(osp.join(self.database.root, relative_image_path))
                image_lst.append(image)
            self.obs = self.APIs["extract_visual_attributes_by_llm"](
                attribute_lst=attribute_lst, image_lst=image_lst
            )
        elif action_name == "FINISH":
            print(f"{action_params=}")
            self.answer = action_params["final_reranked_answer_list"]
            self.obs = "Finished. Answer: {}".format(self.answer)
            done = True
        else:
            self.obs = "Invalid action: {}".format(action_name)
            self.steps += 1

        return self.obs, obs_img, reward, done, self._get_info()

    def get_time_info(self):
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }


class React(ModelForQA):

    def __init__(
        self,
        database,
        llm_func_model,
        emb_model,
        output_dir,
        chunk_size,
        node_emb_dir,
        query_emb_dir,
        chunk_emb_dir,
        n_init_candidates=20,
        n_limit=100,
        dataset="amazon",
        vision=False
    ):
        """
        Answer the query by GPT model.
        Args:
            database (src.benchmarks.semistruct.SemiStruct): database
            organization (str): openai organization
            api_key (str): openai api_key
            model_name (str): model name
            mode (str): one of ['selection', 'yes/no', 'generation']
        """

        super(React, self).__init__(database)
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_emb_dir = chunk_emb_dir
        self.node_emb_dir = node_emb_dir
        self.llm_func_model = llm_func_model
        self.emb_model = emb_model
        self.query_emb_dir = query_emb_dir
        self.n_init_candidates = n_init_candidates
        self.n_limit = n_limit
        self.database = database
        self.dataset = dataset
        self.vision = vision
        self.debug_print_dir = osp.join(output_dir, "debug_print")
        self.debug_print_path = osp.join(self.debug_print_dir, f"{os.getpid()}.txt")
        os.makedirs(self.debug_print_dir, exist_ok=True)
        self.parent_vss = VSS(database, query_emb_dir, node_emb_dir)

    def step(self, env, action, action_param):
        attempts = 0
        while attempts < 3:
            try:
                return env.step(action, action_param)
            except requests.exceptions.Timeout:
                attempts += 1
                print(f"Timeout error, retrying {action} with {action_param} for {attempts} times")

    def extract_parameter_info_without_defaults(self, func_signature):
        func_name = func_signature[:func_signature.find("(")].strip()
        params_block = func_signature[func_signature.find("(") + 1:func_signature.rfind(")")].strip()
        params_block_modified = re.sub(r'\[(.*?)\]', lambda x: x.group(0).replace(',', ';'), params_block)
        params = [param.strip().replace(';', ',') for param in params_block_modified.split(',') if param]
        param_info = {}
        for param in params:
            if ':' in param:
                parts = param.split(':')
                param_name = parts[0].strip()
                # Extracting type and ignoring default value if present
                param_type = parts[1].split('=')[0].strip()
                param_info[param_name] = param_type
            else:
                param_info[param.strip()] = 'Unknown type'
        return func_name, param_info

    def extract_function_details(self, function_signature, model_name):
        # params = re.findall(r"(\w+): (\w+)", function_signature)
        func_name, param_info = self.extract_parameter_info_without_defaults(function_signature)
        param_dict = {param_name: {} for param_name, param_type in param_info.items()}
        if 'gpt' in model_name:
            type_descriptions = [f"{param_name} should be {param_type}" for param_name, param_type in param_info.items()]
            type_des = ", ".join(type_descriptions)
        elif 'claude' in model_name:
            type_des = {}
            mappings = {
                    'int': "integer",
                    'str': "string",
                    'torch.Tensor': "integer"
                }
            for param_name, param_type in param_info.items():
                if param_type in ['List[str]', 'Union[str, List[str]]', 'List[PIL.Image.Image]']:
                    type_des[param_name] = {'type': 'array', 'items': {"type": "string"}, 'description': " ", "minItems": 1}
                elif param_type in ['List[int]', 'List[float]', 'Union[int, List[int]]', 'torch.FloatTensor']:
                    type_des[param_name] = {'type': 'array', 'items': {"type": "integer"}, 'description': " ", "minItems": 1}
                elif param_type in ['int', 'str', 'torch.Tensor']:
                    param_type = mappings.get(param_type, param_type)
                    type_des[param_name] = {'type': param_type, 'description': " "}
                elif param_type == 'List[Dict[str, str]]':
                    type_des[param_name] = {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'additionalProperties': {
                                'type': 'string'
                            }
                        },
                        'description': " ",
                        'minItems': 1
                    }
                else:
                    raise ValueError(f"Type {param_type} not supported")
        else:
            raise ValueError(f"Model {model_name} not supported")    
        return func_name, param_dict, type_des

    def get_initial_prompt(self, env, model_name, in_context_examples=None, n_examples=5):
        if 'gpt' in model_name:
            tool_path = 'avatar/tools/react/tool_lists_gpt.json'
            prompt, tool_list = self._get_prompt_gpt(
                env=env, 
                tool_path=tool_path, 
                model_name=model_name,
                dataset=self.dataset
            )
        elif 'claude' in model_name:
            if self.vision:
                tool_path = 'avatar/tools/react/tool_lists_vision.json'
            else:
                tool_path = 'avatar/tools/react/tool_lists.json'
            prompt, tool_list = self._get_prompt_claude(
                env=env, 
                tool_path=tool_path, 
                model_name=model_name,
                dataset=self.dataset,
                in_context_examples=in_context_examples
            )
        else:
            raise ValueError(f"Model {model_name} not supported")
        return prompt, tool_list

    def _get_prompt_gpt(self, env, tool_path, model_name, dataset, **kwargs):
        if self.vision:
            prompt_path = "avatar/prompts/react_prompt_gpt_flickr.txt"
            prompt = open(prompt_path, "r").read()
        else:
            prompt_path = "avatar/prompts/react_prompt_gpt.txt"
            prompt = open(prompt_path, "r").read()
            prompt = prompt.replace("<node_types>", str(self.database.node_type_lst()))
            prompt = prompt.replace("<edge_types>", str(self.database.rel_type_lst()))
            prompt = prompt.replace("<relational_tuples>", str(self.database.get_tuples()))
            prompt = prompt.replace("<node_attr_dict>", str(self.database.node_attr_dict))
        print(f"In total {len(env.APIs.values())} API tools")
        if os.path.exists(tool_path):
            tool_list = json.load(open(tool_path, "r"))
            return prompt, tool_list
        tool_list = []
        for func in env.APIs.values():
            func_dict = {}
            func_signature = func.func_format.split("->")[0]
            print(f"{func_signature=}")
            function_name, param_dict, type_description = self.extract_function_details(func_signature, model_name)
            func_dict["name"] = function_name
            func_dict["description"] = func.description
            input_schema_dict = {}
            input_schema_dict["type"] = "object"
            input_schema_dict["properties"] = type_description
            input_schema_dict["required"] = list(type_description.keys())
            func_dict["input_schema"] = input_schema_dict
            tool_list.append(func_dict)
        if not os.path.exists(tool_path):
            json.dump(tool_list, open(tool_path, "w"), indent=4)
        dataset_description = ''
        if self.dataset == 'amazon':
            dataset_description = "amazon products"
        elif self.dataset == 'mag':
            dataset_description = "microsoft academic graph"
        elif self.dataset == 'primekg':
            dataset_description = "biomedical knowledge graph" 
        elif self.dataset == 'flickr30k_entities':
            dataset_description = "flickr30k image text dataset"
        prompt = prompt.replace("<dataset_description>", dataset_description)
        print(f'{dataset_description=}')
        return prompt, tool_list
    
    def _get_prompt_claude(self, env, tool_path, model_name, dataset, in_context_examples=None, **kwargs):
        if dataset == 'flickr30k_entities':
            prompt_path = "avatar/prompts/react_prompt_claude_flickr.txt"
            prompt = open(prompt_path, "r").read()
        else:
            if in_context_examples is not None:
                prompt_path = "avatar/prompts/react_prompt_claude_in_context.txt"
                prompt = open(prompt_path, "r").read()
                prompt = prompt.replace("<in_context_examples>", str(in_context_examples))
            else:
                prompt_path = "avatar/prompts/react_prompt_claude.txt"
            prompt = open(prompt_path, "r").read()
            prompt = prompt.replace("<node_types>", str(self.database.node_type_lst()))
            prompt = prompt.replace("<edge_types>", str(self.database.rel_type_lst()))
            prompt = prompt.replace("<relational_tuples>", str(self.database.get_tuples()))
            prompt = prompt.replace("<node_attr_dict>", str(self.database.node_attr_dict))

        print(f"In total {len(env.APIs.values())} API tools")
        if os.path.exists(tool_path):
            tool_list = json.load(open(tool_path, "r"))
            return prompt, tool_list
        tool_list = []
        print('Please handwrite the tool list parameters description for now!')
        for func in env.APIs.values():
            func_dict = {}
            func_signature = func.__str__().split("->")[0]
            print(f"{func_signature=}")
            function_name, param_dict, type_description = self.extract_function_details(func_signature, model_name)
            func_dict["name"] = function_name
            func_dict["description"] = func.description
            input_schema_dict = {}
            input_schema_dict["type"] = "object"
            input_schema_dict["properties"] = type_description
            input_schema_dict["required"] = list(type_description.keys())
            func_dict["input_schema"] = input_schema_dict
            tool_list.append(func_dict)
        if not os.path.exists(tool_path):
            json.dump(tool_list, open(tool_path, "w"), indent=4)
            print(f"Tool list saved to {tool_path}!")
        dataset_description = ''
        if self.dataset == 'amazon':
            dataset_description = "amazon products"
        elif self.dataset == 'mag':
            dataset_description = "microsoft academic graph"
        elif self.dataset == 'primekg':
            dataset_description = "biomedical knowledge graph" 
        elif self.dataset == 'flickr30k_entities':
            dataset_description = "flickr30k image text dataset"
        prompt = prompt.replace("<dataset_description>", dataset_description)
        print(f'{dataset_description=}')
        return prompt, tool_list

    def extract_integers(self, text):
        numbers = re.findall(r'\d+', text)
        return [int(number) for number in numbers]
    
    def claude_execute(self, env, contents, thought_action, prompt):
        done = False
        final_answers = None
        user_execution_feedback = []
        info = {}
        if thought_action['stop_reason'] == 'tool_use':
            print('Use function calling')
            for content in contents:
                if content['type']  == 'text':
                    thinking = content['text']
                elif content['type'] == 'tool_use':
                    func_name = content['name']
                    func_id = content['id']
                    func_input = content['input']
                    obs_text, obs_img, r, done, info = self.step(env, func_name, func_input)
                    if obs_img is not None:
                        image_media_type = "image/jpeg"
                        execution_feed_back = {"type": "tool_result", "tool_use_id": func_id, "content": [{"type": "image", "source": {"type": "base64", "media_type": image_media_type,"data": obs_img}}]}
                    elif obs_text is not None:
                        execution_feed_back = {"type": "tool_result", "tool_use_id": func_id, "content": obs_text}
                    user_execution_feedback.append(execution_feed_back)
                else:
                    raise ValueError(f"Content type {content['type']} not supported")
        elif thought_action['stop_reason'] == 'end_turn' or thought_action['stop_reason'] == 'stop_sequence':
            print('Finish the reasoning, output final answer')
            for content in contents:
                if content['type'] == 'text':
                    output = content['text']
                    try:
                        final_answers = eval(output)
                        done = True
                        break
                    except:
                        raise ValueError(f"Output {output} not a list. Retry")
                else:
                    continue
            print(f'{final_answers=}')
        elif thought_action['stop_reason'] == 'max_tokens':
            print('Exceeds max tokens, retry')
            prompt += 'Exceeds max tokens, retry'
        else:
            raise ValueError(f"Stop reason {thought_action['stop_reason']} not supported")
        return user_execution_feedback, final_answers, done, prompt, info
    
    def gpt_execute(self, env, thought_action, prompt):
        done = False
        final_answers = None
        user_execution_feedback = []
        info = {}
        if thought_action.finish_reason == 'tool_calls':
            print('Use function calling')
            func_name = thought_action.message.tool_calls[0].function.name
            func_input = thought_action.message.tool_calls[0].function.arguments
            func_input = json.loads(func_input)
            func_id = thought_action.message.tool_calls[0].id
            obs, r, done, info = self.step(env, func_name, func_input)
            execution_feed_back = {"role": "function", "tool_call_id": func_id, "name": func_name, "content": obs}
            user_execution_feedback.append(execution_feed_back)
        elif thought_action.finish_reason == 'stop':
            print('Finish the reasoning, output final answer')
            output = thought_action.message.content
            try:
                final_answers = eval(output)
                done = True
            except:
                raise ValueError(f"Output {output} not a list. Retry")
            print(f'{final_answers=}')
        elif thought_action.finish_reason == 'length':
            print('Exceeds max tokens, retry')
            prompt += 'Exceeds max tokens, retry'
        elif thought_action.finish_reason == 'null':
            print('output not yet finished')
            prompt += 'Output is incomplete, retry'
        else:
            raise ValueError(f"Stop reason {thought_action['stop_reason']} not supported")
        return user_execution_feedback, final_answers, done, prompt, info

    def react_think(self, env, question, prompt, vss_top_candidates, model_name, tool_list, in_context_examples=None, key_insights=None, to_print=False, max_think=8, max_call=30):
        print("Start react thinking...")
        print(f"{model_name=}")
        print(f"{question=}")
        print(f"{vss_top_candidates=}")
        prompt = prompt.replace('<question>', question)
        prompt = prompt.replace('<candidates>', str(vss_top_candidates))
        prompt = prompt.replace('<max_think>', str(max_think))
        if in_context_examples is not None:
            prompt = prompt.replace('<in_context_examples>', str(in_context_examples))
            prompt = prompt.replace('<key_insights>', str(key_insights))
        n_calls, n_badcalls = 0, 0
        history = None
        done = False
        fail_flag = 0
        step = 0
        r = 0.0
        info = {}
        for i in range(1, max_think):
            step += 1
            n_calls += 1
            if 'claude' in model_name:
                # print(f'{prompt=}')
                thought_action = get_llm_output_tools(
                    prompt, tools=tool_list, model=model_name, history=history, return_raw=True, json_object=True
                )
                print(f'{thought_action=}')
                thought_action = thought_action.to_dict()
                contents = thought_action["content"]
                user_execution_feedback = []
                try:
                    user_execution_feedback, final_answers, done, prompt, info = self.claude_execute(env, contents, thought_action, prompt)
                except:
                    print('execution failed')
                    prompt += 'Last execution failed! Please retry'
                    step -= 1
                    if n_calls > max_call:
                        print('Exceeds max call limit! Failed to analyze the question')
                        break
                    continue

                if step == 1:
                    history = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": str(contents)},
                    ]
                else:
                    history.append({"role": "user", "content": prompt})
                    history.append({"role": "assistant", "content": str(contents)})
                    
                prompt = str(user_execution_feedback)

            else:
                assert 'gpt' in model_name, f"Model {model_name} not supported"

            if done:
                final_answers = env.answer
                break

        if not done:
            print("NOT FOUND!")
            print(f"Failed to find the answer in {max_think} steps, print original list")
            final_answers = vss_top_candidates
            fail_flag = 1
        if to_print:
            print(info, "\n")
        info.update({"n_calls": n_calls, "n_badcalls": n_badcalls, "traj": prompt})
        return done, final_answers, r, info, history, fail_flag

    def forward(self, query, query_id=None, in_context_examples=None, key_insights=None, **kwargs: Any):
        print("Start forward")
        fail_flag = 0
        env = ReactEnv(
            database=self.database,
            chunk_size=self.chunk_size,
            chunk_emb_dir=self.chunk_emb_dir,
            node_emb_dir=self.node_emb_dir,
            model_name=self.llm_func_model,
            emb_model=self.emb_model,
            debug_print_path=self.debug_print_path,
            dataset=self.dataset,
            n_limit=100,
            initial_temperature=0.2,
            n_init_candidates=self.n_init_candidates,
            use_chunk=False,
        )
        env.reset()
        print("React env setup done")
        prompt, tool_list = self.get_initial_prompt(env, model_name=self.llm_func_model, in_context_examples=in_context_examples)
        print("Prompt setup done")
        # print(f"{prompt=}")
        initial_score_dict, vss_top_candidates = self.get_parent_topk(
            query, query_id, topk=self.n_init_candidates
        )
        succ, final_answers, r, info, history, fail_flag = self.react_think(
            env, 
            query, 
            prompt, 
            vss_top_candidates=vss_top_candidates, 
            model_name=self.llm_func_model,
            tool_list=tool_list,
            in_context_examples=in_context_examples,
            key_insights=key_insights)
        
        print("Found the answer:", final_answers)
        # print(f'{type(final_answers)=}')
        if isinstance(final_answers, str):
            try:
                final_answers = eval(final_answers)
            except:
                print('Failed to parse final answer')
                final_answers = []
        if final_answers is None or len(final_answers) == 0:
            fail_flag = 1
            final_answers = vss_top_candidates 
        final_answers = list(final_answers)
        for i in range(len(final_answers)):
            final_answers[i] = int(final_answers[i])
        f_ans = len(final_answers)
        pred_dict = {}
        for id, ans in enumerate(final_answers):
            pred_dict[ans] = (f_ans - id) / f_ans
        return pred_dict, fail_flag, history


    def _output_to_code(self, output, time_limit=None):
        """
        Extract the code from the output of the code generator.
        """
        assert (
            len(output.split("```")) == 3
        ), "The output should contain only one code block wrapped by ```!"
        code = output.split("```")[1]
        # remove 'python'
        if code.startswith("python"):
            code = code[len("python") :]

        # extract the main function get_node_score_dict
        str_func = "def get_node_score_dict"
        assert str_func in output, f"The output should contain function {str_func}!"

        # add time limit
        if time_limit:
            code = code.replace(str_func, f"@exit_after({time_limit})\n" + str_func)
        return code

    def _exec_code_from_output(self, output, time_limit=None):
        """
        Execute the code from the output of the code generator.
        """
        fail_exec = False
        code = None
        try:
            code = self._output_to_code(output, time_limit)
            exec(code, globals())
        except Exception as err:
            if code:
                string_exec_error_handler(err, code)
            else:
                traceback.print_exc()
        if globals().get("get_node_score_dict") is None:
            fail_exec = True
        if globals().get("parameter_dict") is None:
            fail_exec = True
        return fail_exec, code

    def get_parent_topk(self, query, query_id, topk=100):
        if query_id is None:
            query_emb = get_openai_embedding(query)
        else:
            query_emb_path = osp.join(self.query_emb_dir, f"query_{query_id}.pt")
            if os.path.exists(query_emb_path):
                query_emb = torch.load(query_emb_path).view(1, -1)
            else:
                query_emb = get_openai_embedding(query)
                torch.save(query_emb, query_emb_path)

        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # get the ids with top k highest scores
        top_k_idx = (
            torch.topk(
                torch.FloatTensor(node_scores), min(topk, len(node_scores)), dim=-1
            )
            .indices.view(-1)
            .tolist()
        )

        vss_top_candidates = [node_ids[i] for i in top_k_idx]
        return initial_score_dict, vss_top_candidates

