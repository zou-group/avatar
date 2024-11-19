import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='amazon', 
                        choices=['amazon', 'prime', 'mag', 'flickr30k_entities'])
    parser.add_argument('--split', default='test', choices=["train", "val", "test", "human_generated_eval"])
    parser.add_argument('--group_idx', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='avatar', choices=["avatar", "VSS", "MultiVSS", "LLMReranker", "LLMvReranker", "React"])

    # for vss and multivss
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument("--multi_vss_topk", type=int, default=None)
    parser.add_argument('--aggregate', type=str, default='max')

    # avatar
    parser.add_argument("--emb_model", type=str, default="text-embedding-ada-002")
    parser.add_argument('--agent_llm', default='gpt-4o')
    parser.add_argument('--api_func_llm', default='gpt-4o')
    parser.add_argument('--num_processes', default=None, type=int)
    parser.add_argument('--topk_test', default=None, type=int)
    parser.add_argument('--topk_eval', type=int, default=None)
    parser.add_argument('--n_eval', type=int, default=None)
    parser.add_argument('--n_examples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_total_steps', type=int, default=None)
    parser.add_argument('--use_group', action='store_true')

    # path
    parser.add_argument('--root_dir', default='/dfs/project/kgrlm/benchmark')
    parser.add_argument('--emb_dir', default='emb/')
    parser.add_argument('--output_dir', default='output/')

    # for eval 
    parser.add_argument("--test_ratio", type=float, default=1.0)

    # for baselines
    # LLMReranker specific settings
    parser.add_argument("--llm_model", type=str, default="gpt-4-1106-preview", help='the LLM to rerank candidates.')
    # LLMvReranker specific settings
    parser.add_argument("--vlm_model", type=str, default="gpt-4-1106-preview", help='the VLM to rerank candidates.')
    parser.add_argument("--llm_topk", type=int, default=10)
    parser.add_argument("--max_retry", type=int, default=3)
    # React specific settings
    parser.add_argument("--n_init_candidates", type=int, default=20, help='the number of candidates to rerank.')
    parser.add_argument("--vision", type=bool, default=False, help='whether or not include vision input')
    # Prediction saving settings
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--save_topk", type=int, default=500, help="topk predicted indices to save")

    # for eval_avatar_by_indices
    parser.add_argument("--topk", default=None, type=int, help="topk candidates")
    parser.add_argument("--output", default=None, help="action strings")
    parser.add_argument("--eval_parameter_dict", default=None, help="parameter_dict strings")
    parser.add_argument("--metrics", default=None, help="metrics to evaluate")
    parser.add_argument("--save_path", default=None, help="save path")
    parser.add_argument("--chunk_indices_path", default=None, help="chunk indices path")
    parser.add_argument("--chunk_emb_dir", default=None, help="chunk emb dir")
    parser.add_argument("--query_emb_dir", default=None, help="query emb dir")
    parser.add_argument("--node_emb_dir", default=None, help="node emb dir")

    return parser.parse_args()

def load_default_args(dataset, json_file='config/default_args.json'):
    with open(json_file, 'r') as f:
        default_args = json.load(f)
    return default_args.get(dataset, {})

def update_args_with_defaults(args, defaults):
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

def parse_args_w_defaults(json_file):
    args = parse_args()
    defaults = load_default_args(args.dataset, json_file)
    update_args_with_defaults(args, defaults)
    return args
