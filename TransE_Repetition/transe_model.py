import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0, p_norm=1):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, positive_triplets, negative_triplets):
        pos_h = self.entity_embeddings(positive_triplets[:, 0])
        pos_r = self.relation_embeddings(positive_triplets[:, 1])
        pos_t = self.entity_embeddings(positive_triplets[:, 2])

        neg_h = self.entity_embeddings(negative_triplets[:, 0])
        neg_r = self.relation_embeddings(negative_triplets[:, 1])
        neg_t = self.entity_embeddings(negative_triplets[:, 2])

        pos_score = torch.norm(pos_h + pos_r - pos_t, p=self.p_norm, dim=1)
        neg_score = torch.norm(neg_h + neg_r - neg_t, p=self.p_norm, dim=1)

        return pos_score, neg_score

    def loss(self, pos_score, neg_score):
        return torch.mean(F.relu(self.margin + pos_score - neg_score))
