import os
import os.path as osp
import random
import sys
import argparse

import torch
from tqdm import tqdm

sys.path.append('.')
from avatar.kb import Flickr30kEntities
from avatar.tools import GetCLIPImageEmbedding, GetCLIPTextEmbedding
from avatar.qa_datasets import QADataset
from stark_qa import load_skb, load_qa
from stark_qa.tools.api import get_openai_embeddings


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and embedding model selection
    parser.add_argument('--dataset', default='amazon', choices=['amazon', 'prime', 'mag', 'flickr30k_entities'])
    parser.add_argument('--flickr30k_entities_root', default=None)
    parser.add_argument('--emb_model', default='text-embedding-ada-002', 
                        choices=[
                            'text-embedding-ada-002', 
                            'text-embedding-3-small', 
                            'text-embedding-3-large',
                            'openai/clip-vit-large-patch14'
                            ]
                        )

    # Mode settings
    parser.add_argument('--mode', default='doc', choices=['doc', 'query', 'image'])

    # Path settings
    parser.add_argument("--emb_dir", default="emb/", type=str)

    # Text settings
    parser.add_argument('--add_rel', action='store_true', default=False, help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, help='make the text compact when input to the model')

    # Evaluation settings
    parser.add_argument("--human_generated_eval", action="store_true", help="if mode is `query`, then generating query embeddings on human generated evaluation split")

    # Batch and node settings
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--n_max_nodes", default=10, type=int)

    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()
    mode_surfix = '_human_generated_eval' if args.human_generated_eval and args.mode == 'query' else ''
    mode_surfix += '_no_rel' if not args.add_rel else ''
    mode_surfix += '_no_compact' if not args.compact else ''
    if args.dataset == 'flickr30k_entities':
        mode_surfix = ''
    emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'{args.mode}{mode_surfix}')
    print(f'Embedding directory: {emb_dir}')
    os.makedirs(emb_dir, exist_ok=True)

    if args.dataset == 'flickr30k_entities':
        if args.mode == 'image':
            encoder = GetCLIPImageEmbedding(model_name=args.emb_model, batch_size=args.batch_size)
            kb = Flickr30kEntities(args.flickr30k_entities_root)
            lst = kb.candidate_ids
            emb_path = osp.join(emb_dir, f'candidate_emb_dict.pt')
        if args.mode == 'query':
            encoder = GetCLIPTextEmbedding(model_name=args.emb_model, batch_size=args.batch_size)
            qa_dataset = QADataset(args.dataset, root=args.flickr30k_entities_root)
            lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
            emb_path = osp.join(emb_dir, f'query_emb_dict.pt')
    else:
        if args.mode == 'doc':
            kb = load_skb(args.dataset)
            lst = kb.candidate_ids
            emb_path = osp.join(emb_dir, f'candidate_emb_dict.pt')
        if args.mode == 'query':
            qa_dataset = load_qa(args.dataset, human_generated_eval=args.human_generated_eval)
            lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
            emb_path = osp.join(emb_dir, f'query_emb_dict.pt')
    random.shuffle(lst)
            
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        exisiting_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        exisiting_indices = []
    remaining_indices = list(set(lst) - set(exisiting_indices))

    items, indices = [], []
    for idx in tqdm(remaining_indices):
        if args.mode == 'query':
            item = qa_dataset.get_query_by_qid(idx)
        elif args.mode == 'doc':
            item = kb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact)
        if args.mode == 'image':
            item = kb.get_image(idx)
        items.append(item)
        indices.append(idx)
        
    print(f'Generating embeddings for {len(items)} items...')
    for i in tqdm(range(0, len(items), args.batch_size)):
        batch_items = items[i:i+args.batch_size]
        if args.dataset == 'flickr30k_entities':
            batch_embs = encoder(batch_items)
        else:
            batch_embs = get_openai_embeddings(
                batch_items, 
                model=args.emb_model, 
                n_max_nodes=args.n_max_nodes
                )
        batch_embs = batch_embs.view(len(batch_items), -1).cpu()
        batch_indices = indices[i:i+args.batch_size]
        for idx, emb in zip(batch_indices, batch_embs):
            emb_dict[idx] = emb
        
    torch.save(emb_dict, emb_path)
    print(f'Saved {len(emb_dict)} embeddings to {emb_path}!')
