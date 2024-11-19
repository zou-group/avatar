import sys
sys.path.append(".")
import json
import argparse
import traceback
import os.path as osp

# Import custom modules
import stark_qa
from avatar.models import get_model
from avatar.kb import Flickr30kEntities
from avatar.qa_datasets import QADataset, STaRKDataset
from scripts.args import parse_args_w_defaults

def split_dataset_indices(total_size, num_chunks):
    """
    Split the dataset indices into chunks for parallel processing.
    
    Args:
        total_size (int): Total number of items in the dataset.
        num_chunks (int): Number of chunks to split the dataset into.
    
    Returns:
        list: List of range objects representing the indices for each chunk.
    """
    chunk_size = total_size // num_chunks
    return [
        range(i * chunk_size, min((i + 1) * chunk_size, total_size))
        for i in range(num_chunks)
    ]

def eval_actions_worker(
    args,
    query_indices,
    metrics,
    output,
    eval_parameter_dict,
    use_group,
    group_idx,
    split,
    topk,
    n_eval,
    save_path,
):
    """
    Evaluate actions for a subset of the dataset.
    
    Args:
        args: Parsed command-line arguments.
        query_indices: Indices of queries to evaluate.
        metrics: Evaluation metrics to use.
        output: Output format.
        eval_parameter_dict: Dictionary of evaluation parameters.
        use_group: Whether to use grouping.
        group_idx: Group index.
        split: Data split to use.
        topk: Top-k value for evaluation.
        n_eval: Number of evaluations.
        save_path: Path to save results.
    
    Returns:
        tuple: Evaluation metrics and CSV results.
    """
    # Load the appropriate dataset and knowledge base
    if args.dataset == "flickr30k_entities":
        kb = Flickr30kEntities(root=args.root_dir)
        qa_dataset = QADataset(args.dataset, root=args.root_dir)
    else:
        kb = stark_qa.load_skb(args.dataset, root=args.root_dir)
        qa_dataset = STaRKDataset(args.dataset, root=args.root_dir)
    
    # Initialize the model
    model = get_model(args, kb)
    model.parent_pred_path = osp.join(
        args.output_dir, 
        f"eval/{args.dataset}/VSS/{args.emb_model}/eval_results_test.csv"
    )
    
    # Ensure query indices are provided
    assert query_indices is not None, "Query indices must be provided"
    
    # Perform sequential evaluation of actions
    eval_metrics, eval_csv = model.sequential_eval_actions(
        qa_dataset,
        metrics,
        output,
        eval_parameter_dict,
        use_group,
        group_idx,
        split=split,
        topk=topk,
        n_eval=n_eval,
        save_path=save_path,
        query_indices=query_indices,
    )
    
    return eval_metrics, eval_csv

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args_w_defaults('config/default_args.json')
    
    # Load query indices from file
    with open(args.chunk_indices_path, "r") as f:
        query_indices = json.load(f)
    
    try:
        # Perform evaluation
        eval_metrics, eval_csv = eval_actions_worker(
            args,
            query_indices,
            json.loads(args.metrics),
            args.output,
            json.loads(args.eval_parameter_dict),
            args.use_group,
            args.group_idx,
            args.split,
            args.topk,
            args.n_eval,
            save_path=args.save_path,
        )
    except Exception as e:
        # Print full traceback in case of an error
        traceback.print_exc()
        print(f"Error occurred: {e}")
