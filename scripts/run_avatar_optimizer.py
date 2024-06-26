import sys
sys.path.append('.')

import os
import os.path as osp

import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

import torch
import stark_qa
from avatar.models import get_model
from stark_qa.tools.seed import set_seed
from avatar.kb import Flickr30kEntities
from avatar.qa_datasets import QADataset
from scripts.args import parse_args_w_defaults
    

if __name__ == '__main__':
    args = parse_args_w_defaults('config/default_args.json')
    set_seed(args.seed)
    
    if args.dataset in ['amazon', 'mag', 'prime']:
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.query_emb_dir = osp.join(emb_root, 'query')
        args.node_emb_dir = osp.join(emb_root, 'doc')
        args.chunk_emb_dir = osp.join(emb_root, 'chunk')
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)
        os.makedirs(args.chunk_emb_dir, exist_ok=True)

        kb = stark_qa.load_skb(args.dataset)
        qa_dataset = stark_qa.load_qa(args.dataset)

    elif  args.dataset == 'flickr30k_entities':
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.chunk_emb_dir = None
        args.query_emb_dir = osp.join(emb_root, 'query')
        args.node_emb_dir = osp.join(emb_root, 'image')
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)

        kb = Flickr30kEntities(root=args.root_dir)
        qa_dataset = QADataset(name=args.dataset, root=args.root_dir)
    
    model = get_model(args, kb)
    model.parent_pred_path = osp.join(args.output_dir, f'eval/{args.dataset}/VSS/{args.emb_model}/eval_results_test.csv')
    
    ################### Generate codes ##################
    if args.dataset in ['amazon', 'mag', 'prime']:
        group = model.generate_group(qa_dataset, batch_size=5, n_init_examples=200, split='train')
        group = model.generate_group(qa_dataset, batch_size=5, split='val')
        group = model.generate_group(qa_dataset, batch_size=5, split='test')

    metrics=['mrr', 'map', 'rprecision',
            'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100',
            'hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@20', 'hit@50'
            ]
    model.optimize_actions(qa_dataset=qa_dataset, 
                           seed=args.seed, 
                           group_idx=args.group_idx, 
                           use_group=args.use_group,
                           n_eval=args.n_eval,
                           n_examples=args.n_examples,
                           n_total_steps=args.n_total_steps,
                           topk_eval=args.topk_eval,
                           topk_test=args.topk_test,
                           batch_size=args.batch_size,
                           metrics=metrics)