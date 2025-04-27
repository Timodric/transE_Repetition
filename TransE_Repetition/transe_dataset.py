import torch
import random
from torch.utils.data import Dataset

class TransEDataset(Dataset):
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.num_entities = num_entities
        self.triple_set = set(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        pos_sample = torch.LongTensor([h, r, t])

        corrupt_head = random.random() < 0.5  #变换实体概率
        while True:
            corrupt_entity = random.randint(0, self.num_entities - 1)
            if corrupt_head:
                if (corrupt_entity, r, t) not in self.triple_set:
                    neg_sample = torch.LongTensor([corrupt_entity, r, t])
                    break
            else:
                if (h, r, corrupt_entity) not in self.triple_set:
                    neg_sample = torch.LongTensor([h, r, corrupt_entity])
                    break

        return pos_sample, neg_sample
