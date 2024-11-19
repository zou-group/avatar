import sys
sys.path.append('.')

import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import stark_qa
from avatar.models import get_model
from scripts.args import parse_args_w_defaults


if __name__ == "__main__":
    args = parse_args_w_defaults("config/default_args.json")
    
    if args.dataset in ['amazon', 'mag', 'prime']:
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.node_emb_dir = osp.join(emb_root, 'doc')
        args.chunk_emb_dir = osp.join(emb_root, 'chunk')
        query_emb_surfix = f'_{args.split}' if args.split == 'human_generated_eval' else ''
        args.query_emb_dir = osp.join(emb_root, f"query{query_emb_surfix}")
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)
        os.makedirs(args.chunk_emb_dir, exist_ok=True)

        kb = stark_qa.load_skb(args.dataset)
        qa_dataset = stark_qa.load_qa(args.dataset, human_generated_eval=args.split=='human_generated_eval')
        surfix = args.llm_model if args.model == 'LLMReranker' else args.emb_model

    elif args.dataset == 'flickr30k_entities':
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.chunk_emb_dir = None
        args.node_emb_dir = osp.join(emb_root, 'image')
        args.query_emb_dir = osp.join(emb_root, 'query')

        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)

        kb = Flickr30kEntities(root=args.root_dir)
        qa_dataset = get_qa_dataset(name=args.dataset, root=args.root_dir)
        surfix = args.vlm_model if args.model == 'LLMvReranker' else args.emb_model

    output_dir = osp.join(args.output_dir, "eval", args.dataset, args.model, surfix)
    os.makedirs(output_dir, exist_ok=True)
    json.dump(vars(args), open(osp.join(output_dir, "args.json"), "w"), indent=4)

    eval_csv_path = osp.join(output_dir, f"eval_results_{args.split}.csv")
    final_eval_path = (
        osp.join(output_dir, f"eval_metrics_{args.split}.json")
        if args.test_ratio == 1.0
        else osp.join(output_dir, f"eval_metrics_{args.split}_{args.test_ratio}.json")
    )

    model = get_model(args, kb)
    split_idx = qa_dataset.get_idx_split(test_ratio=args.test_ratio)

    eval_metrics = [
        "mrr",
        "map",
        "rprecision",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
    ]
    if 'React' in args.model:
        eval_csv = pd.DataFrame(columns=['idx', 'query_id', 'pred_rank'] + eval_metrics + ['fail_flag'])
    else:
        eval_csv = pd.DataFrame(columns=['idx', 'query_id', 'pred_rank'] + eval_metrics)
    
    existing_idx = []
    if osp.exists(eval_csv_path):
        eval_csv = pd.read_csv(eval_csv_path)
        existing_idx = eval_csv["idx"].tolist()

    indices = split_idx[args.split].tolist()
    remaining_indices = set(indices) - set(existing_idx)

    for idx in tqdm(remaining_indices):
        query, query_id, answer_ids, meta_info = qa_dataset[idx]
        kwargs = {"seed": args.seed, "split": args.split} if args.model == "avatar" else {}
        if 'React' in args.model:
            pred_dict, fail_flag, history = model.forward(query, query_id, **kwargs)
        else:
            pred_dict = model.forward(query, query_id, **kwargs)

        answer_ids = torch.LongTensor(answer_ids)
        result = model.evaluate(pred_dict, answer_ids, metrics=eval_metrics)

        result["idx"], result["query_id"] = idx, query_id
        result["pred_rank"] = torch.LongTensor(list(pred_dict.keys()))[
            torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[
                :args.save_topk
            ]
        ].tolist()
        if 'React' in args.model:
            result['fail_flag'] = fail_flag

        eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)

        if args.save_pred:
            eval_csv.to_csv(eval_csv_path, index=False)
        for metric in eval_metrics:
            print(
                f"{metric}: {np.mean(eval_csv[eval_csv['idx'].isin(indices)][metric])}"
            )
    if args.save_pred:
        eval_csv.to_csv(eval_csv_path, index=False)
    final_metrics = (
        eval_csv[eval_csv["idx"].isin(indices)][eval_metrics].mean().to_dict()
    )
    json.dump(final_metrics, open(final_eval_path, "w"), indent=4)