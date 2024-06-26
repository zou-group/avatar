import copy
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import re
import time
import torch
import traceback
from functools import partial
from datetime import datetime
from subprocess import Popen
from typing import Any, Union, List, Dict
from tqdm import tqdm

from avatar.tools import assigned_funcs, customized_funcs, general_funcs
from avatar.models.model import ModelForQA
from avatar.models.vss import VSS
from avatar.utils.device import auto_select_device
from avatar.utils.error_handler import string_exec_error_handler
from stark_qa.tools.io import read_from_file, write_to_file
from avatar.utils.timer import exit_after
from stark_qa.skb import SKB
from stark_qa.tools.api import get_llm_output


class MemoryBank:
    def __init__(self, memory_types: List[str], file_path: str = None):
        self.memory_types = memory_types
        if file_path:
            self.load_from_json(file_path)
        else:
            for memory_type in memory_types:
                setattr(self, memory_type, [])

    def jsonable(self) -> Dict[str, List]:
        return {memory_type: getattr(self, memory_type) for memory_type in self.memory_types}
    
    def push(self, memory_type: str, memory: Any) -> None:
        mem = getattr(self, memory_type)
        setattr(self, memory_type, mem + [memory])
    
    def pop(self, memory_type: str) -> Any:
        mem = getattr(self, memory_type)
        last_mem = mem[-1]
        setattr(self, memory_type, mem[:-1])
        return last_mem
    
    def load_from_json(self, path: str) -> None:
        memory_bank = read_from_file(path)
        for memory_type in self.memory_types:
            setattr(self, memory_type, memory_bank[memory_type])


class AvaTaR(ModelForQA): 
  
    def __init__(self, 
                 kb: Any,
                 emb_model: str,
                 agent_llm: str,
                 api_func_llm: str,
                 output_dir: str,
                 chunk_size: int,
                 node_emb_dir: str,
                 query_emb_dir: str,
                 chunk_emb_dir: str,
                 threshold: float = 0.5,
                 n_limit: int = 50,
                 topk_test: int = 200,
                 num_processes: int = 4,
                 dataset: str = 'amazon',
                 time_limit_unit: int = 20
                 ):
        """
        Initialize the AvaTaR class.

        Args:
            kb (Any): The knowledge base object.
            agent_llm (str): The model name or path for the actions generator.
            api_func_llm (str): The model name or path for the LLM function model.
            output_dir (str): The directory where outputs will be saved.
            chunk_size (int): The size of text chunks for processing.
            node_emb_dir (str): The directory where node embeddings are stored.
            query_emb_dir (str): The directory where query embeddings are stored.
            chunk_emb_dir (str): The directory where chunk embeddings are stored.
            n_limit (int, optional): The maximum number of iterations or calls. Default is 50.
            topk_test (int, optional): The number of initial candidates to consider. Default is 200.
            num_processes (int, optional): The number of processes to use for parallel processing. Default is 4.
            dataset (str, optional): The name of the dataset being used. Default is 'amazon'.
            time_limit_unit (int, optional): The time limit unit to constrain the execution time 
        """

        super().__init__(kb=kb)
        # Initialize class variables
        self.kb = kb
        self.emb_model = emb_model
        self.agent_llm = agent_llm
        self.api_func_llm = api_func_llm
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_emb_dir = chunk_emb_dir
        self.query_emb_dir = query_emb_dir
        self.node_emb_dir = node_emb_dir
        self.n_limit = n_limit
        self.threshold = threshold
        self.topk_test = topk_test
        self.num_processes = num_processes
        self.dataset = dataset
        self.time_limit_unit = time_limit_unit

        ###########################################################
        #                    Modulize components                  # 
        ###########################################################
        # preprocessor for grouping queries (only once)
        self.preprocessor = partial(get_llm_output, model=self.api_func_llm, 
                                    json_object=True, max_tokens=4096, temperature=0.5)
        # actor for producing actions
        self.actor = partial(get_llm_output, model=self.agent_llm, temperature=1)
        # comparator for generating instructions for the actor
        self.comparator = partial(get_llm_output, model=self.agent_llm, temperature=1)

        # Initialize parent VSS model
        self.parent_pred_path = None
        self.parent_vss = VSS(kb, query_emb_dir, node_emb_dir, emb_model=emb_model)

        # Set up debug print paths
        self.debug_print_dir = osp.join(output_dir, 'debug_print')
        self.debug_print_path = osp.join(self.debug_print_dir, f'{os.getpid()}.txt')
        os.makedirs(self.debug_print_dir, exist_ok=True)

        # Initialize APIs
        self.APIs = self._get_APIs()
    
    def _load_actions(self, group_idx: int, seed: int = 20) -> Union[str, Dict]:
        group_output_dir = osp.join(self.output_dir, f'group_{group_idx}', f'seed_{seed}')
        actions_best_path = osp.join(group_output_dir, f'actions_best.txt')
        param_best_path = osp.join(group_output_dir, 'actions_best_param.json')
        
        actions_best = read_from_file(actions_best_path)
        param_best = read_from_file(param_best_path)
        return actions_best, param_best
    
    def _get_APIs(self) -> Dict[str, Any]:
        assigned_funcs_key = self.dataset if self.dataset not in ['amazon', 'prime', 'mag'] else 'stark'
        apis = assigned_funcs[assigned_funcs_key]
        available_funcs = general_funcs
        if self.dataset in customized_funcs.keys():
            available_funcs.update(customized_funcs[self.dataset])

        kwargs_union = {
            'kb': self.kb,
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

    def _get_prompt(self, name: str = 'initialize_actions', **kwargs: Any) -> str:
        
        prompt_path = {
            'initialize_actions_stark': 'prompts/avatar_initialize_actions_stark.txt',
            'initialize_actions_flickr30k_ent': 'prompts/avatar_initialize_actions_flickr30k_ent.txt',
            'improve_actions': 'prompts/avatar_improve_actions.txt',
            'comparator': 'prompts/avatar_comparator.txt',
            'assign_group': 'prompts/preprocess_group_assignment.txt',
            'initialize_group': 'prompts/preprocess_group_initialization.txt'
        }
        current_dir = osp.dirname(osp.abspath(__file__))
        prompt_path = {key: osp.join(current_dir, '..', path) \
            for key, path in prompt_path.items()}

        if name == 'comparator':
            prompt = read_from_file(prompt_path[name])
            pos_neg_queries = kwargs['pos_neg_queries']
            prompt = prompt.replace('<pos_neg_queries>', pos_neg_queries)

        if name == 'initialize_group':
            prompt = read_from_file(prompt_path[name])
            example_queries = kwargs['example_queries']
            prompt = prompt.replace('<node_types>', str(self.kb.node_type_lst()))
            prompt = prompt.replace('<edge_types>', str(self.kb.rel_type_lst()))
            prompt = prompt.replace('<relational_tuples>', str(self.kb.get_tuples()))
            prompt = prompt.replace('<node_attr_dict>', str(self.kb.node_attr_dict))
            prompt = prompt.replace('<example_queries>', example_queries)

        elif name == 'initialize_actions':
            if isinstance(self.kb, SKB):
                prompt = read_from_file(prompt_path['initialize_actions_stark'])
                sample_indices = kwargs['sample_indices']
                qa_dataset = kwargs['qa_dataset']
                pattern = kwargs['pattern']
                func_call_description = '\n'.join(['- ' + func.func_format + '. ' + func.description for func in self.funcs])
                example_queries = '\n'.join([f'Q{i+1}: ' + qa_dataset[idx][0] for i, idx in enumerate(sample_indices)])
                prompt = prompt.replace('<node_types>', str(self.kb.node_type_lst()))
                prompt = prompt.replace('<edge_types>', str(self.kb.rel_type_lst()))
                prompt = prompt.replace('<relational_tuples>', str(self.kb.get_tuples()))
                prompt = prompt.replace('<node_attr_dict>', str(self.kb.node_attr_dict))
                prompt = prompt.replace('<topk_test>', str(self.topk_test))
                prompt = prompt.replace('<func_call_description>', func_call_description)
                prompt = prompt.replace('<candidate_types>', 
                                        str(set([self.kb.get_node_type_by_id(node_id) for node_id in self.kb.candidate_ids])))
                prompt = prompt.replace('<pattern>', pattern)
                prompt = prompt.replace('<example_queries>', example_queries)
            elif self.dataset == 'flickr30k_entities':
                prompt = read_from_file(prompt_path['initialize_actions_flickr30k_ent'])
                sample_indices = kwargs['sample_indices']
                qa_dataset = kwargs['qa_dataset']
                pattern = kwargs['pattern']
                func_call_description = '\n'.join(['- ' + func.func_format + '. ' + func.description for func in self.funcs])
                example_queries = '\n'.join([f'Q{i+1}: ' + qa_dataset[idx][0] for i, idx in enumerate(sample_indices)])

                prompt = prompt.replace('<topk_test>', str(self.topk_test))
                prompt = prompt.replace('<func_call_description>', func_call_description)
                prompt = prompt.replace('<pattern>', pattern)
                prompt = prompt.replace('<example_queries>', example_queries)

        elif name == 'improve_actions':
            prompt = read_from_file(prompt_path[name])
            debug_message = kwargs['debug_message']
            feedback_message = kwargs['feedback_message']
            query = kwargs['query']
            candidate_ids = kwargs['candidate_ids']
            debug_message = debug_message.strip(' \n')
            prompt = prompt.replace('<debug_message>', '"\n' + debug_message + '\n"' if len(debug_message) else 'No output')
            prompt = prompt.replace('<feedback_message>', feedback_message)
            prompt = prompt.replace('<input_query>', '"' + query + '"' if len(query) else 'Not specified')
            prompt = prompt.replace('<size_of_candidate_ids>', str(len(candidate_ids)) if len(candidate_ids) else 'Not specified')

        elif name == 'assign_group':
            prompt = read_from_file(prompt_path[name])
            query = kwargs['query']
            group_patterns = kwargs['group_patterns']
            prompt = prompt.replace('<query>', query)
            prompt = prompt.replace('<group_patterns>', group_patterns)

        return prompt

    def _parse_output_to_actions(self, output: str, time_limit: int = None) -> str:
        '''
        Extract the actions from the output of the actions generator.
        '''
        assert len(output.split('```')) == 3, 'The output should contain only one actions block wrapped by a pair of "```"!'
        actions = output.split('```')[1]
        # remove 'python'
        if actions.startswith('python'):
            actions = actions[len('python'):]

        # extract the main function get_node_score_dict
        str_func = 'def get_node_score_dict'
        # assert str_func in output, f'The output should contain function {str_func}!'

        # add time limit
        if time_limit:
            actions = actions.replace(str_func, f"@exit_after({time_limit})\n" + str_func)
        else:
            # remove “@exit_after(xxx)"
            actions = re.sub(r"@exit_after\(\d+\)", "", actions)
        return actions

    def _exec_actions_from_output(self, output: str, time_limit: int = None) -> Union[bool, str, str]:
        '''
        Execute the actions from the output of the actions generator.
        '''
        fail_exec = False
        fail_exec_info = None
        actions = None
        try:
            actions = self._parse_output_to_actions(output, time_limit)
            exec(actions, globals())
        except Exception as err:
            if actions:
                fail_exec_info = string_exec_error_handler(err, actions)
            else:
                traceback.print_exc()
                fail_exec_info = traceback.format_exc()
            print(fail_exec_info)
        if globals().get('parameter_dict') is None or \
           globals().get('get_node_score_dict') is None:
            fail_exec = True
        return fail_exec, actions, fail_exec_info

    def optimize_actions(self, 
                         qa_dataset: Any, 
                         use_group: bool = True, 
                         group_idx: int = 0, 
                         seed: int = 0, 
                         n_examples: int = 25, 
                         n_total_steps: int = 200, 
                         n_eval: int = 100, 
                         batch_size: int = 20, 
                         topk_eval: int = 30, 
                         patience: int = 10,
                         topk_test: int = 30, 
                         metrics: List[str] = ['hit@5', 'recall@20'], 
                         sel_metric: str = 'MRR', 
                         verbose: bool = True):
        
        ###########################################################
        #            Initialize global state and APIs             #
        ###########################################################
        def freeze_global() -> List[str]:
            return copy.deepcopy(list(globals().keys()))

        def reset_APIs() -> None:
            globals().update(self.APIs)

        def reset_global(reset_keys: List[str]) -> None:
            cur_keys = copy.copy(list(globals().keys()))
            reset_keys = list(set(cur_keys) - set(reset_keys))
            for key in reset_keys:
                globals().pop(key)
            if len(reset_keys) > 0:
                print(f'Remove keys {reset_keys} in globals()')
            reset_APIs()

        # load group assignment
        if use_group:
            group_train, patterns = self.load_group(surfix='train')
            group_val, patterns = self.load_group(surfix='val')
            train_indices = group_train[group_idx]['query_idx'] + group_val[group_idx]['query_idx']
            pattern = group_train[group_idx]['pattern']
        else:
            train_indices = qa_dataset.get_idx_split()['train'].tolist()
            pattern = 'NA'
        random.shuffle(train_indices)

        # prepare for the output directory
        print(f'Generating for seed {seed}...')
        if use_group:
            output_dir = osp.join(self.output_dir, f'group_{group_idx}')
        else:
            output_dir = self.output_dir
        actions_output_dir = osp.join(output_dir, f'seed_{seed}')
        os.makedirs(actions_output_dir, exist_ok=True)

        log_path = osp.join(actions_output_dir, 'log.json')
        initial_actions_path = osp.join(actions_output_dir, 'actions_initial.txt')
        actions_curr_path = osp.join(actions_output_dir, 'actions_curr.txt')
        actions_best_path = osp.join(actions_output_dir, 'actions_best.txt')
        actions_best_param_path = osp.join(actions_output_dir, 'actions_best_param.json')
        actions_best_metric_path = osp.join(actions_output_dir, 'actions_best_metric.json')
        memory_bank_path = osp.join(actions_output_dir, 'memory_bank.json')
        metadata_path = osp.join(actions_output_dir, f'metadata.json')
        
        # initialize variables
        global parameter_dict
        global get_node_score_dict
        
        ###########################################################
        #                     Initialize actions                  #
        ###########################################################
        def get_initial_prompt() -> str:
            sample_indices = random.sample(train_indices, min(n_examples, len(train_indices)))
            prompt = self._get_prompt(
                name='initialize_actions', 
                sample_indices=sample_indices,
                qa_dataset=qa_dataset, 
                pattern=pattern)
            return prompt
                
        def initialize_actions() -> Union[str, str]:
            prompt = get_initial_prompt()
            output = self.actor(prompt)
            return output, prompt

        memory_bank = MemoryBank(['action_performance', 'supervison_info'])
        if not osp.exists(actions_best_metric_path):
            step, best_step, gap_from_last_improv = 0, 0, 0
            best_metric = {sel_metric.lower(): -1}
            best_param_dict, best_output = None, None
            if osp.exists(initial_actions_path):
                prompt = get_initial_prompt()
                output = read_from_file(initial_actions_path)
            else:
                ###########################################################
                #                 Generate initial actions                #
                ###########################################################
                output, prompt = initialize_actions()
                write_to_file(initial_actions_path, output)
            if verbose:
                print('\nprompt:\n', prompt)
                print('\noutput:\n', output)
            curr_log = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": output}]
        else:
            ###########################################################
            #                        Resume                           #
            ###########################################################
            metadata = read_from_file(metadata_path)
            step, best_step = metadata['step'], metadata['best_step']
            gap_from_last_improv = metadata['gap_from_last_improv']
            best_metric = read_from_file(actions_best_metric_path)
            best_output = read_from_file(actions_best_path)
            best_param_dict = read_from_file(actions_best_param_path)
            memory_bank.load_from_json(memory_bank_path)
            curr_log = read_from_file(log_path)
            output = read_from_file(actions_curr_path)
            assert curr_log[-1]['content'] == output, f'Bad resume! Check the log at {actions_curr_path}'
            
        ###########################################################
        #            Executability testing and self-improving     #
        ###########################################################
        freezed_before_improv = freeze_global()
        while step < n_total_steps:
            comparator_instruction = None
            query, candidate_ids = '', []
            try:
                exec_eval = {}
                added_superv_to_mem = False
                reset_global(freezed_before_improv)
                fail_exec, actions, fail_exec_info = self._exec_actions_from_output(output, time_limit=self.time_limit_unit * topk_eval)
                reset_APIs()
                
                assert not fail_exec, fail_exec_info
                assert globals().get('parameter_dict') is not None, '`parameter_dict` is not defined!'
                assert globals().get('get_node_score_dict') is not None, '`get_node_score_dict` is not defined!'
                
                sampled_batch, debug_messages = [], []
                random.shuffle(train_indices)
                for idx in train_indices:
                    query, query_id, answer_ids, meta_info = qa_dataset[idx]
                    _, candidate_ids = self.get_parent_topk(query, query_id, topk=topk_eval)
                    vss_cnt = len(set(candidate_ids).intersection(set(answer_ids)))
                    if vss_cnt == 0: 
                        continue
                    
                    self.APIs['debug_print'].clean_file()
                    self.APIs['debug_print'].enable()
                    node_score_dict = get_node_score_dict(query, candidate_ids, **parameter_dict)
                    debug_message = self.APIs['debug_print'].get_written()
                    debug_messages.append(debug_message)
                    
                    scores = torch.FloatTensor(list(node_score_dict.values()))
                    if verbose:
                        print('parameter_dict', parameter_dict)
                        print('node_score_dict', node_score_dict)

                    ###########################################################
                    #                Check the format of the output           #
                    ###########################################################
                    assert len(node_score_dict) == len(candidate_ids), f'The length of node_score_dict {len(node_score_dict)} is not equal to the length of candidate_ids {len(candidate_ids)}!'
                    assert scores.numel() == len(candidate_ids), f'The number of scores {scores.numel()} is not equal to the length of candidate_ids {len(candidate_ids)}!'
                    assert all([isinstance(v, float) or isinstance(v, int) for v in node_score_dict.values()]), f'The values of node_score_dict {node_score_dict} should be float or int!'
                    assert len(set(scores.tolist())) > 1 or len(node_score_dict) == 1, f'The scores in node_score_dict {node_score_dict} are all the same! Please avoid trivial solutions!'

                    ###########################################################
                    #             Evaluate the actions on the query              #
                    ###########################################################
                    exec_eval[idx] = self.evaluate(
                        node_score_dict, torch.LongTensor(answer_ids), 
                        metrics=['hit@1', 'hit@5', 'recall@20', 'mrr']
                        )
                    sampled_batch.append(idx)
                    if len(sampled_batch) >= (1.5 * batch_size):
                        break
                    torch.cuda.empty_cache()
                    
                self.APIs['debug_print'].clean_file()
                self.APIs['debug_print'].enable()
                self.APIs['debug_print'](''.join(debug_messages))
                self.APIs['debug_print'].disable()

                query, candidate_ids = '', []
                ###########################################################
                #              Construct pos & neg queries                #
                ###########################################################
                pos_neg_queries = self.construct_pos_neg_queries(
                    qa_dataset, sampled_batch, exec_eval, batch_size, sel_metric, self.threshold
                    )
                comparator_prompt = self._get_prompt(name='comparator', pos_neg_queries=pos_neg_queries)
                
                kb_schema_prompt = curr_log[0]['content']
                last_actions = curr_log[-1]['content']
                comparator_instruction = self.comparator([
                    {"role": "user", "content": kb_schema_prompt},
                    {"role": "assistant", "content": '```\n' + last_actions + '\n```'},
                    {"role": "user", "content": comparator_prompt}]
                )
                added_superv_to_mem = True
                memory_bank.push('supervison_info', pos_neg_queries)
                write_to_file(memory_bank_path, memory_bank.jsonable())
                
                ###########################################################
                #               Evaluate the improved actions             # 
                #       in the last step (before mem bank increase)       #
                ###########################################################
                try:
                    last_best_metric = copy.copy(best_metric)
                    save_path = osp.join(actions_output_dir, f'eval_action_step{step}.json')
                    actions_per_step_path = osp.join(actions_output_dir, f'actions_step{step}.txt')
                    write_to_file(actions_per_step_path, output)
                    out_param, out_metric = self.eval_action(
                            output, qa_dataset, metrics,
                            use_group, group_idx, 
                            save_path=save_path, topk=topk_eval, 
                            n_eval=n_eval
                            )
                    if out_metric[sel_metric.lower()] > best_metric[sel_metric.lower()]:
                        best_metric, best_param_dict, best_output = out_metric, out_param, output
                        write_to_file(actions_best_param_path, best_param_dict)
                        write_to_file(actions_best_metric_path, best_metric)
                        write_to_file(actions_best_path, best_output)
                    actions = self._parse_output_to_actions(output)
                    memory_bank.push('action_performance', (actions, out_metric))
                    
                except Exception as err:
                    fail_exec_info = traceback.format_exc()
                    print(f'Fail to execute the improved actions! {fail_exec_info}')
                    pass
                
                if best_metric[sel_metric.lower()] > last_best_metric[sel_metric.lower()]:
                    best_step = step
                    gap_from_last_improv = 0
                else:
                    gap_from_last_improv += 1

                # Raise exception at the end of this step
                assert False, comparator_instruction

            except Exception as err:
                ###########################################################
                #                     Handle the error                    #
                ###########################################################
                exec_eval = {}
                kb_schema_prompt = curr_log[0]['content']
                last_actions = curr_log[-1]['content']
                
                if comparator_instruction:
                    feedback_message = comparator_instruction
                else:
                    feedback_message = string_exec_error_handler(err, actions)
                debug_message = self.APIs['debug_print'].get_written().strip(' \n')
                improve_actions_prompt = self._get_prompt(name='improve_actions',
                                                            feedback_message=feedback_message,
                                                            debug_message=debug_message,
                                                            query=query,
                                                            candidate_ids=candidate_ids)

                ###########################################################
                #                 Format memory bank info                 #
                ###########################################################
                memory_info, last_actions_metric = '', ''
                if len(memory_bank.action_performance):
                    action_performance = memory_bank.action_performance
                    actions_mem = [perf[0] for perf in action_performance if perf[0] != last_actions]
                    metrics_mem = [perf[1] for perf in action_performance if perf[0] != last_actions]
                    metric_mem = [perf[1][sel_metric.lower()] for perf in action_performance if perf[0] != last_actions]
                    topk_actions_indices = np.argsort(metric_mem)[::-1][:3].tolist()
                    actions_mem = [actions_mem[i] for i in topk_actions_indices]
                    metrics_mem = [metrics_mem[i] for i in topk_actions_indices]
                    
                    last_metric = None
                    for c, m in action_performance:
                        if c == last_actions:
                            last_metric = m
                            break
                    
                    memory_info = f'The following information stores your memory to help you generate code better.\n' + \
                                    f'These are the previous generated codes and their evaluation metrics on the validation queries:\n' + \
                                    '\n'.join([f'#{i + 1}:\n' + 'code:\n' + '```python\n' + c + '```\n' + 'Metrics:\n' + 
                                    '  Hit@1: ' + str(round(m["hit@1"], 4)) + '\n' + 
                                    '  Hit@5: ' + str(round(m["hit@5"], 4)) + '\n' + 
                                    '  Recall@20: ' + str(round(m["recall@20"], 4)) + '\n' + 
                                    '  MRR: ' + str(round(m["mrr"], 4)) + '\n' for i, (c, m) in enumerate(zip(actions_mem, metrics_mem))])
                    if last_metric:
                        last_actions_metric = (f'By executing the code in your last message, the evaluation metrics on validation queries are:\n' + 
                                                '  Hit@1: ' + str(last_metric["hit@1"]) + '\n' + 
                                                '  Hit@5: ' + str(last_metric["hit@5"]) + '\n' + 
                                                '  Recall@20: ' + str(m["recall@20"]) + '\n' + 
                                                '  MRR: ' + str(last_metric["mrr"]) + '\n')
                error_handle_log = [{"role": "user", "content": kb_schema_prompt + '\n' + memory_info}]
                error_handle_log.append({"role": "assistant", "content": '```\n' + last_actions + '\n```'})
                error_handle_log.append({"role": "user", "content": last_actions_metric + '\n' + improve_actions_prompt})

                ########################################################### 
                #                  Obtain improved program                #
                #               or try reinitializing the actions         #
                ###########################################################
                if not added_superv_to_mem or gap_from_last_improv <= patience:
                    output = self.actor(error_handle_log)
                    if verbose:
                        print(improve_actions_prompt)
                        print(output)
                    curr_log.append({"role": "user", "content": improve_actions_prompt})
                    curr_log.append({"role": "assistant", "content": output})
                else:
                    gap_from_last_improv = 0
                    output, prompt = initialize_actions()
                    curr_log.append({"role": "user", "content": prompt})
                    curr_log.append({"role": "user", "content": output})
                    if verbose:
                        print(prompt)
                        print(output)
                write_to_file(log_path, curr_log)
                write_to_file(actions_curr_path, output)

            step += 1
            metadata = {'step': step, 'best_step': best_step, 
                        'gap_from_last_improv': gap_from_last_improv,
                        'time': str(datetime.now())}
            write_to_file(metadata_path, metadata)
            self.APIs['debug_print'].clean_file()

            if step % 25 == 0 or step == n_total_steps:
                test_save_path = osp.join(actions_output_dir, f'eval_metrics_test_topk{topk_test}_step{step}.json')
                ###########################################################
                #             Evalulation on the Testing dataset          #
                ###########################################################
                try:
                    best_param_dict = read_from_file(actions_best_param_path)
                    best_output = read_from_file(actions_best_path)
                except Exception as err:
                    print('No successful actions found!')
                    continue
                if not osp.exists(test_save_path):
                    print('############## Eval ##############')
                    actions_best_step_path = osp.join(actions_output_dir, f'actions_best_step{step}.txt')
                    actions_best_param_step_path = osp.join(actions_output_dir, f'actions_best_param_step{step}.json')
                    write_to_file(actions_best_step_path, best_output)
                    write_to_file(actions_best_param_step_path, best_param_dict)
                    eval_metrics, _ = self.parallel_eval_actions(self.dataset, qa_dataset, metrics,
                                                                 best_output, best_param_dict, 
                                                                 use_group, group_idx, 
                                                                 split='test', topk=topk_test, 
                                                                 n_eval=-1, save_path=test_save_path,
                                                                 num_processes=self.num_processes
                                                                 )
                    print(eval_metrics)
                    
    def construct_pos_neg_queries(self, qa_dataset, 
                                  batch, exec_eval, 
                                  batch_size, sel_metric, threshold):

        sorted_idx = np.argsort([exec_eval[idx][sel_metric.lower()] for idx in batch])[::-1]
        sorted_idx = np.array(batch)[sorted_idx]
        sorted_queries = [qa_dataset[idx][0] for idx in sorted_idx]
        sorted_metric = [{'hit@1': exec_eval[idx]['hit@1'], 
                            'hit@5': exec_eval[idx]['hit@5'],
                            'recall@20': exec_eval[idx]['recall@20'], 
                            'mrr': exec_eval[idx]['mrr']} for idx in sorted_idx
                            ]
        pos_queries = [f'Query {i + 1}: {sorted_queries[i]}\n=>' + 
                        '  Hit@1: ' + str(round(m['hit@1'], 4)) + 
                        '  Hit@5: ' + str(round(m['hit@5'], 4)) + 
                        '  Recall@20: ' + str(round(m['recall@20'], 3)) + 
                        '  MRR: ' + str(round(m['mrr'], 4)) + '\n' for i, m in enumerate(sorted_metric) if m['hit@5'] > self.threshold]
        neg_queries = [f'Query {i + 1}: {sorted_queries[i]}\n=>' + 
                        '  Hit@1: ' + str(round(m['hit@1'], 4)) + 
                        '  Hit@5: ' + str(round(m['hit@5'], 4)) + 
                        '  Recall@20: ' + str(round(m['recall@20'], 3)) + 
                        '  MRR: ' + str(round(m['mrr'], 4)) + '\n' for i, m in enumerate(sorted_metric) if m['hit@5'] <= self.threshold]
        if len(neg_queries) and len(pos_queries):
            neg_queries = neg_queries[-batch_size // 2:]
            pos_queries = pos_queries[:batch_size // 2]
        else:
            queries = pos_queries + neg_queries
            pos_queries = queries[:len(queries) // 2]
            neg_queries = queries[len(queries) // 2:]
        pos_neg_queries = ['Positive examples:\n'] + pos_queries  + \
                           ['Negative examples:\n'] + neg_queries
        pos_neg_queries = '\n'.join(pos_neg_queries)
        return pos_neg_queries
        
    def eval_action(self, 
                    output: str, 
                    qa_dataset: Any, 
                    metrics: List[str],
                    use_group: bool, 
                    group_idx: int, 
                    save_path: str, 
                    topk: int = 50, 
                    n_eval: int = 50) -> Union[Dict[str, Any], Dict[str, float]]:

        if osp.exists(save_path):
            param_best = read_from_file(save_path)
            return param_best['param'], param_best['metric']

        globals().update(self.APIs)
        fail_exec, actions, _ = self._exec_actions_from_output(output, time_limit=None)
        globals().update(self.APIs)
        if fail_exec: 
            print('Abort! Fail to execute the actions!')
            return

        parameter_dict = globals().get('parameter_dict')
        
        param_search_eval = {}
        search_eval, eval_csv = self.parallel_eval_actions(
            self.dataset, qa_dataset, metrics, 
            output, parameter_dict, 
            use_group, group_idx, 
            split='val', topk=topk, n_eval=n_eval,
            save_path=osp.join(osp.dirname(save_path), f'val_eval_metric.json'),
            num_processes=self.num_processes
            )
        param_search_eval = {'param': parameter_dict, 
                             'metric': search_eval, 
                             'n_eval': len(eval_csv)}
        write_to_file(save_path, param_search_eval)
        return param_search_eval['param'], param_search_eval['metric']
    
        
    def get_eval_indices(self, 
                         qa_dataset: Any, 
                         split: str, 
                         use_group: bool, 
                         group_idx: int, 
                         n_eval: int, 
                         query_indices: List[int] = None) -> List[int]:
        """
        1. Provide query_indices to evaluate on specific queries
        2. Eval on test, if n_eval is -1, evaluate on all queries; otherwise eval on max n_eval queries
        3. Evaluate on val, if n_eval is -1, evaluate on all queries; otherwise eval on max n_eval queries, with supplement from train
        """
        assert split in ['val', 'test']
        if query_indices:
            indices = query_indices
        else:
            if use_group:
                if split == 'val':
                    val_indices = self.load_group(surfix='val')[0][group_idx]['query_idx']
                    if n_eval > 0 and len(val_indices) < n_eval:
                        train_indices = self.load_group(surfix='train')[0][group_idx]['query_idx']
                        random.Random(0).shuffle(train_indices)
                        indices = val_indices + train_indices[:n_eval - len(val_indices)]
                    else:
                        indices = val_indices
                elif split == 'test':
                    indices = self.load_group(surfix='test')[0][group_idx]['query_idx']
            else:
                indices = qa_dataset.get_idx_split()[split].tolist()
        random.Random(0).shuffle(indices)
        indices = indices[:n_eval] if n_eval > 0 else indices
        return indices

    def sequential_eval_actions(self, 
                                qa_dataset: Any, 
                                metrics: List[str],
                                output: str, 
                                eval_parameter_dict: Dict[str, Any], 
                                use_group: bool, 
                                group_idx: int, 
                                split: str, 
                                topk: int, 
                                n_eval: int = -1, 
                                save_path: str = None, 
                                query_indices: List[int] = None) -> Union[Dict[str, float], pd.DataFrame]:
        
        indices = self.get_eval_indices(qa_dataset, split, use_group,
                                        group_idx, n_eval, query_indices)
        if save_path:
            json_save_path = save_path
            file_name = osp.basename(save_path).split('.')[0]
            csv_save_path = osp.join(os.path.dirname(save_path), f'{file_name}.csv')
        save_root = osp.dirname(save_path) if save_path else '.'
    
        globals().update(self.APIs)
        fail_exec, actions, _ = self._exec_actions_from_output(output)
        globals().update(self.APIs)

        parameter_dict = eval_parameter_dict
        if fail_exec:
            import pdb; pdb.set_trace()
        
        eval_metrics = {}
        get_node_score_dict = globals().get('get_node_score_dict')
        eval_csv = pd.DataFrame(columns=['idx', 'query_id', 'pred_rank'] + metrics)
        for idx in tqdm(indices):
            query, query_id, answer_ids, _ = qa_dataset[idx]
            _, candidate_ids = self.get_parent_topk(query, query_id, topk=topk)
            vss_cnt = len(set(candidate_ids).intersection(set(answer_ids)))
            if split == 'val' and vss_cnt == 0: 
                continue
            
            success = False
            for _ in range(3):
                # While it is unlikely to fail during eval, we still need to 
                # handle the error due to api connection error, oom error, etc.
                try:
                    pred_dict = get_node_score_dict(query, candidate_ids, **parameter_dict)
                    success = True
                    break
                except Exception as err:
                    error_message = string_exec_error_handler(err, actions)
                    print(error_message)
                    with open(osp.join(save_root, 'latest_eval_error.log'), 'a+') as f:
                        f.write(error_message)
                    
            if success:
                result = self.evaluate(pred_dict, 
                                       torch.LongTensor(answer_ids), 
                                       metrics=metrics)
                result['idx'], result['query_id'] = idx, query_id
                result['pred_rank'] = torch.LongTensor(list(pred_dict.keys()))[
                    torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[:1000]].tolist()
            else:
                try:
                    result = {'idx': idx, 'query_id': query_id, 'pred_rank': []}
                    result.update({metric: -1 for metric in metrics})
                    with open(osp.join(save_root, 'latest_eval_error.log'), 'a+') as f:
                        f.write(f'Fail to execute on query {query_id}!')
                except Exception as err:
                    import pdb; pdb.set_trace()

            eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)
            if save_path:
                eval_csv.to_csv(path_or_buf=csv_save_path, index=False)
        for metric in metrics:
            eval_metrics[metric] = np.mean([eval_csv[metric].iloc[i] for i in range(len(eval_csv))])
        if save_path:
            write_to_file(json_save_path, eval_metrics)

        return eval_metrics, eval_csv

    def parallel_eval_actions(self, 
                              dataset: str, 
                              qa_dataset: Any, 
                              metrics: List[str], 
                              output: str, 
                              eval_parameter_dict: Dict[str, Any], 
                              use_group: bool, 
                              group_idx: int, 
                              split: str, 
                              topk: int, 
                              n_eval: int = -1, 
                              save_path: str = None, 
                              num_processes: int = 4) -> Union[Dict[str, float], pd.DataFrame]:
        t1 = time.time()
        json_save_path = save_path
        file_name = osp.basename(save_path).split('.')[0]
        csv_save_path = osp.join(os.path.dirname(save_path), f'{file_name}.csv')

        save_dir = osp.dirname(save_path)
        temp_dir = osp.join(save_dir, "parallel_eval")
        os.makedirs(temp_dir, exist_ok=True)
        chunk_indices_pathss = [osp.join(temp_dir, f'indices_chunk_{idx}.json') for idx in range(num_processes)]
        chunk_json_save_paths = [osp.join(temp_dir, f"eval_metrics_chunk_{idx}.json") for idx in range(num_processes)]
        chunk_csv_save_paths = [osp.join(temp_dir, f"eval_metrics_chunk_{idx}.csv") for idx in range(num_processes)]

        eval_indices = self.get_eval_indices(qa_dataset, split, use_group, group_idx, n_eval, query_indices=None)
        total_size = len(eval_indices)
        chunk_ranges = AvaTaR.split_dataset_indices(total_size=total_size, num_chunks=num_processes)
        print(f'Parallel evaluting on {total_size} queries....')

        for chunk_range, chunk_path in zip(chunk_ranges, chunk_indices_pathss):
            query_indices = [eval_indices[idx] for idx in chunk_range]
            write_to_file(chunk_path, query_indices)

        if torch.cuda.is_available():
            try:
                cuda_lst = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
                cuda_lst = [int(cuda) for cuda in cuda_lst]
                devices = [auto_select_device(cuda_lst) for _ in range(num_processes)]
            except:
                devices = ['0' for _ in range(num_processes)]
        else:
            devices = [None for _ in range(num_processes)]

        metrics_arg = json.dumps(metrics)
        eval_parameter_dict_args = json.dumps(eval_parameter_dict)
        
        commands = []
        for chunk_json_save_path, chunk_indices_paths in zip(chunk_json_save_paths, chunk_indices_pathss):
            command = [
                "python", "scripts/eval_avatar_by_indices.py",
                "--dataset", dataset,
                "--split", split,
                "--output", output,
                "--metrics", metrics_arg,
                "--emb_model", self.emb_model, 
                "--api_func_llm", self.api_func_llm,
                "--agent_llm", self.agent_llm,
                "--topk", str(topk),
                "--n_eval", str(n_eval),
                "--eval_parameter_dict", eval_parameter_dict_args,
                "--root_dir", osp.dirname(self.kb.root),
                "--save_path", chunk_json_save_path,
                "--chunk_indices_path", chunk_indices_paths,
                "--chunk_emb_dir", str(self.chunk_emb_dir),
                "--query_emb_dir", str(self.query_emb_dir),
                "--node_emb_dir", str(self.node_emb_dir)
            ]
            if use_group:
                command = command + ["--use_group",
                                     "--group_idx", str(group_idx)]
            commands.append(command)
        
        processes = []
        for device, command in zip(devices, commands):
            str_device = device.split(':')[-1]
            print('CUDA_VISIBLE_DEVICES:', str_device)
            process = Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str_device))
            processes.append(process)

        for process in processes:
            process.wait()

        eval_csvs = []
        for chunk_csv_path in chunk_csv_save_paths:
            eval_csvs.append(pd.read_csv(chunk_csv_path))
        eval_csv = pd.concat(eval_csvs, ignore_index=True)

        eval_csv.to_csv(path_or_buf=csv_save_path, index=False)
        eval_metrics = {}
        for metric in metrics:
            eval_metrics[metric] = np.mean([eval_csv[metric].iloc[i] for i in range(len(eval_csv))])
        write_to_file(json_save_path, eval_metrics)
        t2 = time.time()
        print(f"Parallel evaluation took {t2 - t1} seconds")
        with open('time.log', 'a+') as f:
            f.write(f"Parallel evaluation took {t2 - t1} seconds\nnum_processes: {num_processes}\n\n")

        return eval_metrics, eval_csv

    @staticmethod
    def split_dataset_indices(total_size: int, num_chunks: int) -> List[range]:
        chunk_size = total_size // num_chunks
        return [range(i * chunk_size, min((i + 1) * chunk_size, total_size)) for i in range(num_chunks)]

    def get_parent_topk(self, query: str, query_id: int, topk: int = 100) -> Union[Dict[int, float], List[int]]:
        if self.parent_pred_path and osp.exists(self.parent_pred_path):
            csv = pd.read_csv(self.parent_pred_path)
            csv = csv[['query_id', 'pred_rank']]
            csv = csv[csv['query_id'] == query_id]
            if len(csv):
                pred_rank = eval(csv['pred_rank'].iloc[0])
                initial_score_dict = {node_id: 1. / (rank + 1) for rank, node_id in enumerate(pred_rank)}
                return initial_score_dict, pred_rank[:topk]

        initial_score_dict = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(topk, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()

        vss_top_candidates = [node_ids[i] for i in top_k_idx]
        return initial_score_dict, vss_top_candidates

    def get_group_id(self, query_idx: int, split: str = None) -> int:
        if split is None:
            split = ['train', 'val', 'test']
        else:
            split = [split]
        for surfix in split:
            group, _ = self.load_group(surfix=surfix)
            for i in range(len(group)):
                if query_idx in group[i]['query_idx']:
                    return i
        
        return None

    def save_group(self, group: Dict[int, Any], surfix: str) -> None:
        save_path = osp.join(self.output_dir, '..', f'group_query_{surfix}.json')
        write_to_file(save_path, group)
    
    def load_group(self, surfix: str = 'current') -> Union[Dict[int, Any], str]:
        path = osp.join(self.output_dir, '..', f'group_query_{surfix}.json')
        group = read_from_file(path)
        group = {int(key): group[key] for key in group.keys()}
        patterns = '\n'.join([f'{i}: ' + group[i]["pattern"] for i in range(len(group))])
        return group, patterns

    def initialize_group(self, 
                         qa_dataset: Any = None, 
                         indices: List[int] = None, 
                         add_none: bool = True, 
                         save_to: str = 'current') -> Dict[int, Any]:
        save_path = osp.join(self.output_dir, f'group_initial_{save_to}.json')
        if osp.exists(save_path):
            group = read_from_file(save_path)
        else:
            example_queries = '\n'.join([f'{i}: ' + qa_dataset[idx][0] for i, idx in enumerate(indices)])
            prompt = self._get_prompt(name='initialize_group', example_queries=example_queries)
            group = self.preprocessor(prompt)
            group = json.loads(group)
            group = {int(key): group[key] for key in group.keys()}
            for i in range(len(group)):
                original_query_idx = torch.LongTensor(group[i]['query_idx'])
                group[i]['query_idx'] = torch.LongTensor(indices)[original_query_idx].tolist()

            if add_none:
                group[len(group)] = {'pattern': 'None of the above', 'query_idx': []}
            if save_to:
                self.save_group(group, surfix=save_to)

        return group

    def assign_group(self, 
                     qa_dataset: Any, 
                     indices: List[int], 
                     append_to: str = 'current') -> int:
        query = '\n'.join([f'{i}: ' + qa_dataset[idx][0] for i, idx in enumerate(indices)])
        group, group_patterns = self.load_group(surfix=append_to)
        prompt = self._get_prompt(name='assign_group', query=query, group_patterns=group_patterns)

        while True:
            try:
                output = self.preprocessor(prompt)
                output = json.loads(output)
                output = {int(key): int(output[key]) for key in output.keys()}
                assert set(list(output.keys())) == set([i for i in range(len(indices))])
                assert set(list(output.values())).issubset(set(list(range(len(group)))))
                break
            except:
                import pdb; pdb.set_trace()
        for query_idx, group_id in output.items():
            group[group_id]['query_idx'] = list(set(group[group_id]['query_idx'] + [indices[query_idx]]))
        self.save_group(group, surfix=append_to)
        return group_id

    def generate_group(self, 
                       qa_dataset: Any, 
                       split: str = 'train', 
                       batch_size: int = 100, 
                       n_init_examples: int = 200) -> Dict[int, Any]:

        path_bootstrap = osp.join(self.output_dir, '..', 'group_query_bootstrap.json')
        path_split = osp.join(self.output_dir, '..', f'group_query_{split}.json')

        if osp.exists(path_split):
            return self.load_group(surfix=split)[0]

        if split == 'train':
            ################### Initialize group ##################
            if not osp.exists(path_bootstrap):
                train_indices = qa_dataset.get_idx_split()['train'].tolist()
                indices = random.sample(train_indices, n_init_examples)
                group_initial = self.initialize_group(qa_dataset, indices, add_none=True, save_to='initial')
                self.save_group({key: {'pattern': group_initial[key]['pattern'], 'query_idx': []} for key in group_initial.keys()}, surfix='bootstrap')
                for batch_idx in tqdm(range(np.ceil(len(train_indices) / batch_size).astype(int))):
                    indices = train_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    self.assign_group(qa_dataset, indices, append_to='bootstrap')

            group_initial = self.load_group(surfix='initial')[0]
            group_bootstrap = self.load_group(surfix='bootstrap')[0]
            
            n_overlap, n_total = 0., 0
            for i in range(len(group_bootstrap)):
                n_total += len(group_initial[i]['query_idx'])
                n_overlap += len(set(group_initial[i]['query_idx']).intersection(set(group_bootstrap[i]['query_idx'])))
            print(f'Overlap: {n_overlap} / {n_total} = ', n_overlap / n_total)

            ############# Categorize None of the above again ############
            group_clear_none = copy.deepcopy(group_bootstrap)
            self.save_group(group_clear_none, surfix=split)
            group_clear_none.pop(len(group_clear_none) - 1)
            group_clear_none[len(group_clear_none)] = {'pattern': 'None of the above', 'query_idx': []}
            unassigned_idx = group_bootstrap[len(group_bootstrap) - 1]['query_idx']
            if not osp.exists(path_split):
                if len(unassigned_idx):
                    for batch_idx in tqdm(range(np.ceil(len(unassigned_idx) / batch_size).astype(int))):
                        indices = unassigned_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                        self.assign_group(qa_dataset, indices, append_to=split)
                
                group_split = self.load_group(surfix=split)[0]
                unassigned_idx_last = group_split[len(group_split) - 1]['query_idx']

                ############# Add new groups from None of the above ############
                if len(unassigned_idx_last):
                    group_new = self.initialize_group(qa_dataset, unassigned_idx_last, add_none=False, save_to='new')
                    group_new = {key + len(group_split) - 1: group_new[key] for key in group_new.keys()}
                    group_split.update(group_new)
                self.save_group(group_split, surfix=split)
        else:
            if not osp.exists(path_split):
                group, patterns = self.load_group(surfix='train')
                group_clear = {key: {'pattern': group[key]['pattern'], 'query_idx': []} for key in group.keys()}
                self.save_group(group_clear, surfix=split)
                split_indices = qa_dataset.get_idx_split()[split].tolist()
                for batch_idx in tqdm(range(np.ceil(len(split_indices) / batch_size).astype(int))):
                    indices = split_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    self.assign_group(qa_dataset, indices, append_to=split)
            group_split = self.load_group(surfix=split)[0]
        return group_split

    def forward(self, 
                query: Union[str, List[str]], 
                query_id: int,
                split: str = 'test',
                seed: int = 0,
                **kwargs: Any) -> Dict[int, float]:

        ############## Get prompt for group classification ##############
        group_id = self.get_group_id(query_id, split=split)
        print('group_id', group_id)
        actions_best, param_best = self._load_actions(group_id, seed=seed)

        globals().update(self.APIs)
        fail_exec, actions, _ = self._exec_actions_from_output(actions_best)
        globals().update(self.APIs)
        
        ############## Use VSS to filter ##############
        initial_score_dict, candidate_ids = self.get_parent_topk(query, query_id, topk=self.topk_test)

        get_node_score_dict = globals().get('get_node_score_dict')
        parameter_dict = globals().get('parameter_dict') 

        if fail_exec:
            import pdb; pdb.set_trace()
            return initial_score_dict

        pred_dict = get_node_score_dict(query, candidate_ids, **parameter_dict)
        return pred_dict