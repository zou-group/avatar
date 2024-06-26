import argparse
import json
import os
import os.path as osp
import random
from typing import Any

import torch
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from avatar.models import VSS
from avatar.models.model import ModelForQA
from stark_qa import load_qa, load_skb

torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# Assuming you have a dataset in the format of (query, positive_document, negative_document)
# For simplicity, let's say it's stored as a list of tuples: [(query, pos_doc, neg_doc), ...]
def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='amazon')
    parser.add_argument('--dataset_save_path', default='/dfs/project/kgrlm/data/dense_retrieval/dataset')
    parser.add_argument('--model_save_path', default='/dfs/project/kgrlm/data/dense_retrieval/model')
    parser.add_argument('--negative_sampling', default='hard_negative', choices=['random', 'hard_negative'])
    parser.add_argument('--preprocess_path', default='/dfs/project/kgrlm/data/dense_retrieval/preprocessed')
    
    parser.add_argument('--num_candidate_negatives', default=100, type=int)
    parser.add_argument('--num_hard_negatives', default=20, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    
    parser.add_argument('--emb_dir', default='/dfs/project/kgrlm/embs')
    parser.add_argument('--max_length', default=256, type=int)
    
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--lbd', default=0.1, type=float)
    
    return parser


class RetrievalDataset(Dataset):
    def __init__(self, data, dataset, kb, qa_dataset, tokenizer, max_length=256, preprocess_path=None, num_hard_negatives=20):
        self.data = data
        self.dataset = dataset
        self.kb = kb
        self.qa_dataset = qa_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_path = preprocess_path
        self.num_hard_negatives = num_hard_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preprocess_path is not None and osp.exists(osp.join(self.preprocess_path, f'{self.dataset}', f'{idx}.pt')):
            pass
        else:
            print('Not yet preprocessed. Start preprocessing...')
            self.preprocess()
        load_path = osp.join(self.preprocess_path, f'{self.dataset}', f'{idx}.pt')
        query_enc, pos_doc_enc, neg_doc_encs = torch.load(load_path)
        return query_enc, pos_doc_enc, neg_doc_encs
    
    def preprocess(self):
        base_path = self.preprocess_path + f'/{self.dataset}'
        os.makedirs(base_path, exist_ok=True)
        for idx in tqdm(range(len(self.data))):
            save_path = osp.join(base_path, f'{idx}.pt')
            query_id, pos_doc_id, neg_doc_id_list = self.data[idx]
            neg_doc_id_list = neg_doc_id_list[:self.num_hard_negatives]
            query, _, _, _ = self.qa_dataset[query_id]
            pos_doc = self.kb.get_doc_info(pos_doc_id, add_rel=True, compact=True)
            query_enc = self.tokenizer(query, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
            pos_doc_enc = self.tokenizer(pos_doc, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
            neg_doc_encs = [self.tokenizer(self.kb.get_doc_info(neg_doc_id, add_rel=True, compact=True), max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt") for neg_doc_id in neg_doc_id_list]
            torch.save((query_enc, pos_doc_enc, neg_doc_encs), save_path)
        print('Preprocessing finished')


class RetrievalModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output


class DenseRetrieval(ModelForQA):
    def __init__(self, kb, model_path, doc_enc_dir, query_emb_dir, candidates_emb_dir, candidates_dir, eval_batch_size, num_candidates, dataset='amazon', renew_candidates=False):
        super().__init__(kb)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model.eval()
        self.kb = kb
        self.eval_batch_size = eval_batch_size
        self.doc_enc_dir = osp.join(doc_enc_dir, f'{dataset}')
        self.candidates_dir = osp.join(candidates_dir, f'{dataset}')
        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.num_candidates = num_candidates
        self.renew_candidates = renew_candidates
        self.qa_dataset = load_qa(dataset)

        os.makedirs(doc_enc_dir, exist_ok=True)
        os.makedirs(self.doc_enc_dir, exist_ok=True)
        os.makedirs(candidates_dir, exist_ok=True)
        os.makedirs(self.candidates_dir, exist_ok=True)

        self.vss_candidates_dict = self.get_vss_candidates()

        
    def forward(self, query, query_id=None, **kwargs: Any):
        pred = {}
        print('Start predicting!')
        query_enc = self.tokenizer(query, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
        query_input_ids = query_enc['input_ids']
        query_attention_mask = query_enc['attention_mask']
        query_emb = self.model(query_input_ids, attention_mask=query_attention_mask).pooler_output
        # get candidate document embeddings
        candidate_dic = self.get_candidate_doc_enc(query_id)
        for doc_id, doc_enc in candidate_dic.items():
            doc_emb = self.model(doc_enc['input_ids'], attention_mask=doc_enc['attention_mask']).pooler_output
            # calculate similarity
            similarity = torch.matmul(query_emb, doc_emb.T).squeeze()
            pred[doc_id] = similarity.item()
        return pred
    
    def get_candidate_doc_enc(self, query_id):
        doc_enc_path = osp.join(self.doc_enc_dir, f'{query_id}.pt')
        if osp.exists(doc_enc_path):
            doc_enc_dict = torch.load(doc_enc_path)
            return doc_enc_dict
        else:
            print(f'Generating document embeddings and saving to {doc_enc_path}')
            os.makedirs(self.doc_enc_dir, exist_ok=True)
            doc_enc_dict = {}
            for candidate_doc_id in self.vss_candidates_dict[str(query_id)]:
                doc = self.kb.get_doc_info(candidate_doc_id, add_rel=True, compact=True)
                doc_enc = self.tokenizer(doc, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
                doc_input_ids = doc_enc['input_ids']
                doc_attention_mask = doc_enc['attention_mask']
                doc_enc_dict[candidate_doc_id] = {'input_ids': doc_input_ids, 'attention_mask': doc_attention_mask}
            torch.save(doc_enc_dict, doc_enc_path)
            return doc_enc_dict

    def get_vss_candidates(self):
        def get_top_k_ids(score_dict, k):
            top_k_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            top_k_ids = [item[0] for item in top_k_items]
            return top_k_ids
        
        if osp.exists(osp.join(self.candidates_dir, 'vss_candidates.json')) and not self.renew_candidates:
            print(f'Loading candidates from {osp.join(self.candidates_dir, "vss_candidates.json")}')
            with open(osp.join(self.candidates_dir, 'vss_candidates.json'), 'r') as file:
                vss_candidates = json.load(file)
            return vss_candidates
        else:
            print('Generating vss candidates...')
            vss = VSS(self.kb, self.query_emb_dir, self.candidates_emb_dir)
            vss_candidates = {}
            
            for i, idx in tqdm(enumerate(self.qa_dataset.get_idx_split()['test'])):
                idx = int(idx)
                query, query_id, answer_ids, meta_info = self.qa_dataset[idx]
                cnt = 0
                while cnt < 10: 
                    try:
                        pred = vss(query=query, query_id=query_id)
                        break
                    except:
                        cnt += 1
                        if cnt == 10:
                            print(f'Error in query {query_id}')
                            raise ValueError(f'Error in query {query_id}')
                # get top k highest scores
                top_k_node_ids = get_top_k_ids(pred, self.num_candidates)
                vss_candidates[str(query_id)] = top_k_node_ids
                if i % int(len(self.qa_dataset.get_idx_split()['test']) / 10) == 0 or i == len(self.qa_dataset) - 1:
                    print(f'saving {i} candidates to {osp.join(self.candidates_dir, f"vss_candidates_{i}.pt")}')
                    # save json file
                    with open(osp.join(self.candidates_dir, f'vss_candidates_{i}.json'), 'w') as file:
                        json.dump(vss_candidates, file, indent=4)
            # save json file
            with open(osp.join(self.candidates_dir, 'vss_candidates.json'), 'w') as file:
                json.dump(vss_candidates, file, indent=4)
            return vss_candidates
    
def get_constrast_data(qa_dataset, 
                       kb, name, 
                       negative_sampling, 
                       emb_dir, 
                       dataset_save_path, 
                       num_candidate_negatives):
    contrast_data = []
    if negative_sampling == 'hard_negative':
        print(f"Using hard negative sampling with {num_hard_negatives} hard negatives")
        filename = osp.join(dataset_save_path, f'{name}_hard_neg_data.txt')
        if osp.exists(filename):
            print(f'Loading contrast data from {osp.join(dataset_save_path, f"{filename}")}')
            with open(filename, 'r') as file:
                for line in file:
                    parts = line.strip().split(',', 2) 
                    a, b = int(parts[0]), int(parts[1]) 
                    c_list = list(map(int, parts[2].split(';')))
                    contrast_data.append((a, b, c_list))
        visited_ind = [tup[0] for tup in contrast_data]
        if not osp.exists(filename) or len(visited_ind) == 0 or max(visited_ind) + 1 != len(qa_dataset):
            print(f'generate hard negatives!')
            temp_list = []
            query_emb_dir = osp.join(emb_dir, name, 'query')
            candidates_emb_dir = osp.join(emb_dir, name, 'doc')
            if len(visited_ind) > 0:
                print(f'Current length of processed query data: {max(visited_ind) + 1}')
            else:
                print('No contrast data found')
            vss = VSS(kb, query_emb_dir, candidates_emb_dir)
            print('format: (query index, positive document index, negative document index list)')
            for i, idx in tqdm(enumerate(qa_dataset.get_idx_split()['train'])):
                idx = int(idx)
                if idx in visited_ind:
                    continue
                query, query_id, answer_ids, meta_info = qa_dataset[idx]
                cnt = 0
                while cnt < 10: 
                    try:
                        pred = vss(query=query, query_id=query_id)
                        break
                    except:
                        cnt += 1
                        if cnt == 10:
                            print(f'Error in query {query_id}')
                            raise ValueError(f'Error in query {query_id}')
                # get top k highest scores
                top_k_idx = torch.topk(torch.FloatTensor([v for k, v in pred.items()]),
                                        min(num_candidate_negatives + len(answer_ids), len(pred)),
                                        dim=-1).indices.view(-1).tolist()
                top_k_node_ids = [k for i, k in enumerate(pred.keys()) if i in top_k_idx]
                # exclude the answer ids
                top_k_node_ids = [k for k in top_k_node_ids if k not in answer_ids]
                top_k_node_ids = top_k_node_ids[:num_candidate_negatives]
                for pos_doc in answer_ids:
                    neg_docs = top_k_node_ids
                    contrast_data.append((idx, pos_doc, neg_docs))
                    temp_list.append((idx, pos_doc, neg_docs))
                if i % 100 == 0 or i == len(qa_dataset) - 1:
                    print(f'saving {i} hard negative candidates to {filename}')
                    with open(filename, 'a') as file:
                        for tup in temp_list:
                            idx, pos_doc, neg_docs = tup
                            c_str = ';'.join(map(str, neg_docs))
                            file.write(f"{idx},{pos_doc},{c_str}\n")
                    temp_list = []
            print(f'saving hard negative candidates to {osp.join(dataset_save_path, f"{name}_hard_neg_data.pt")}')
            torch.save(contrast_data, osp.join(dataset_save_path, f'{name}_hard_neg_data.pt'))
        else:
            print(f'Loaded {max(visited_ind) + 1} hard negative candidates')
    else:
        print("Using random negative sampling")
        if osp.exists(osp.join(dataset_save_path, f'{name}_random_data.txt')):
            print(f'Loading random data from {osp.join(dataset_save_path, f"{name}_random_data.txt")}')
            contrast_data = torch.load(osp.join(dataset_save_path, f'{name}_random_data.txt'))
        else:
            print(f'generate random negatives')
            for idx in range(len(qa_dataset)):
                query, query_id, answer_ids, meta_info = qa_dataset[idx]
                random_negs = random.sample(range(len(qa_dataset)), num_candidate_negatives + len(answer_ids))
                random_negs = [n for n in random_negs if n not in answer_ids]
                contrast_data.append((idx, answer_ids, random_negs))
            print(f'len(random_data): {len(contrast_data)}')
            print(f'saving random candidates to {osp.join(dataset_save_path, f"{name}_random_data.pt")}')
            torch.save(contrast_data, osp.join(dataset_save_path, f'{name}_random_data.txt'))
            assert len(contrast_data) == len(qa_dataset)
    return contrast_data


if __name__ == '__main__':

    parser = arg_parse()
    args = parser.parse_args()
    name = args.dataset
    kb = load_skb(name)
    qa_dataset = load_qa(name)
    num_candidate_negatives = args.num_candidate_negatives
    negative_sampling = args.negative_sampling
    num_hard_negatives = args.num_hard_negatives
    epochs = args.epochs
    dataset_save_path = args.dataset_save_path
    model_save_path = args.model_save_path
    emb_dir = args.emb_dir
    max_length = args.max_length
    lbd = args.lbd
    batch_size = args.batch_size
    preprocess_path = args.preprocess_path
    lr = args.learning_rate

    contrast_data = get_constrast_data(qa_dataset, kb, name, negative_sampling, emb_dir, dataset_save_path, num_candidate_negatives)
    print('finished getting contrast data')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    retrieval_dataset = RetrievalDataset(contrast_data, name, kb, qa_dataset, tokenizer, max_length, preprocess_path)
    dataloader = DataLoader(retrieval_dataset, batch_size=batch_size, shuffle=True)
    retrieval_model = RetrievalModel(model)
    retrieval_model.train()
    if torch.cuda.is_available():
        retrieval_model.to('cuda:5')
        retrieval_model = DataParallel(retrieval_model, 
        device_ids=[5])
    # retrieval_model.to('cuda:1')
    device = next(retrieval_model.parameters()).device
    print(f'Using {device} device')

    # Define your loss function and optimizer
    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(retrieval_model.parameters(), lr=lr)

    print('Start training')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        batch_tqdm = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in batch_tqdm:
            query_enc, pos_doc_enc, neg_doc_encs = batch  
            neg_doc_encs = neg_doc_encs[:num_hard_negatives]
            query_input_ids = query_enc['input_ids'].squeeze(1).to(device)
            query_attention_mask = query_enc['attention_mask'].squeeze(1).to(device)
            pos_input_ids = pos_doc_enc['input_ids'].squeeze(1).to(device)
            pos_attention_mask = pos_doc_enc['attention_mask'].squeeze(1).to(device)
            
            neg_input_ids_list = []
            neg_attention_mask_list = []
            for neg_doc_enc in neg_doc_encs:
                neg_input_ids = neg_doc_enc['input_ids'].squeeze(1) 
                neg_attention_mask = neg_doc_enc['attention_mask'].squeeze(1)
                neg_input_ids_list.append(neg_input_ids)
                neg_attention_mask_list.append(neg_attention_mask)
                
            neg_input_ids = torch.cat(neg_input_ids_list, dim=0).to(device)
            neg_attention_mask = torch.cat(neg_attention_mask_list, dim=0).to(device)

            assert query_input_ids.device == query_attention_mask.device == pos_input_ids.device == pos_attention_mask.device == neg_input_ids.device == neg_attention_mask.device
            query_emb = retrieval_model(query_input_ids, attention_mask=query_attention_mask)
            pos_doc_emb = retrieval_model(pos_input_ids, attention_mask=pos_attention_mask)
            neg_doc_emb = retrieval_model(neg_input_ids, attention_mask=neg_attention_mask)
            
            # Positive pairs have a target of 1
            pos_target = torch.ones(pos_doc_emb.size(0)).to(pos_doc_emb.device)
            # Negative pairs have a target of -1
            neg_target = torch.ones(neg_doc_emb.size(0)).to(neg_doc_emb.device) * -1
            
            loss_pos = loss_fn(query_emb, pos_doc_emb, pos_target)
            query_emb_repeated = query_emb.repeat_interleave(neg_doc_emb.size(0) // query_emb.size(0), dim=0)

            loss_neg = loss_fn(query_emb_repeated, neg_doc_emb, neg_target)            
            loss = loss_pos + lbd * loss_neg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_tqdm.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch}, Loss {total_loss / len(dataloader)}")

        save_dir = osp.join(model_save_path, f'{name}_{epoch}_retrieval_model')
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    print('Training finished')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
