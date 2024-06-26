import copy
import os.path as osp
import pandas as pd
import torch
from typing import Dict, List, Tuple, Union


class QADataset:
    def __init__(self, name: str, root: str):
        """
        General QA Dataset class.

        Args:
            name (str): Name of the dataset.
            root (str): Root directory where the dataset is stored.
        """

        self.split_dir = osp.join(root, name, 'split')
        self.qa_csv_path = osp.join(root, name, 'qa.csv')
        print('Loading QA dataset from', self.qa_csv_path)
        self.data = pd.read_csv(self.qa_csv_path)

        self.indices = list(self.data['id'])
        self.indices.sort()
        self.split_indices = self.get_idx_split()
    
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[str, int, List[int], Union[None, str]]:
        """
        Get the query, query ID, answer IDs, and meta info for a given index.

        Args:
            idx (int): Index of the query.

        Returns:
            Tuple[str, int, List[int], Union[None, str]]: Query, query ID, answer IDs, and meta info.
        """
        q_id = self.indices[idx]
        meta_info = None
        row = self.data[self.data['id'] == q_id].iloc[0]
        query = row['query']
        answer_ids = eval(row['answer_ids'])
        
        return query, q_id, answer_ids, meta_info

    def get_idx_split(self, test_ratio: float = 1.0) -> Dict[str, torch.LongTensor]:
        """
        Return the indices of train/val/test split in a dictionary.

        Args:
            test_ratio (float, optional): Ratio of the test split to use. Default is 1.0.

        Returns:
            Dict[str, torch.LongTensor]: Dictionary with train/val/test indices.
        """
        split_idx = {}
        for split in ['train', 'val', 'test']:
            # `{split}.index` stores query ids, not the index in the dataset
            indices_file = osp.join(self.split_dir, f'{split}.index')
            indices = open(indices_file, 'r').read().strip().split('\n')
            query_ids = [int(idx) for idx in indices]
            split_idx[split] = torch.LongTensor([self.indices.index(query_id) for query_id in query_ids])
        if test_ratio < 1.0:
            split_idx['test'] = split_idx['test'][:int(len(split_idx['test']) * test_ratio)]
        return split_idx

    def get_query_by_qid(self, q_id: int) -> str:
        """
        Return the query by query ID.

        Args:
            q_id (int): Query ID.

        Returns:
            str: Query string.
        """
        row = self.data[self.data['id'] == q_id].iloc[0]
        return row['query']
        
    def get_subset(self, split: str) -> 'QADataset':
        """
        Return a subset of the dataset.

        Args:
            split (str): Data split to return (train/val/test).

        Returns:
            QADataset: Subset of the dataset.
        """
        assert split in ['train', 'val', 'test']
        indices_file = osp.join(self.split_dir, f'{split}.index')
        indices = open(indices_file, 'r').read().strip().split('\n')
        subset = copy.deepcopy(self)
        subset.indices = [int(idx) for idx in indices]
        return subset
