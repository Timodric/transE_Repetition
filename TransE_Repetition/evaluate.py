import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def evaluate_transe(model, data, all_triples_set, num_entities, device='cpu', batch_size=128, filtered=True):
    model.eval()
    hits_at_10 = []
    mean_ranks = []

    for h, r, t in tqdm(data, desc="Evaluating"):
        h, r, t = int(h), int(r), int(t)

        # 替换头实体
        head_candidates = torch.arange(num_entities).to(device)
        rel_batch = torch.full_like(head_candidates, r)
        tail_batch = torch.full_like(head_candidates, t)
        triples = torch.stack([head_candidates, rel_batch, tail_batch], dim=1)

        scores = torch.norm(
            model.entity_embeddings(triples[:, 0]) +
            model.relation_embeddings(triples[:, 1]) -
            model.entity_embeddings(triples[:, 2]),
            p=model.p_norm,
            dim=1
        )

        if filtered:
            for corrupt_h in range(num_entities):
                if corrupt_h != h and (corrupt_h, r, t) in all_triples_set:
                    scores[corrupt_h] = float('inf')

        rank = torch.argsort(scores).tolist().index(h) + 1
        mean_ranks.append(rank)
        hits_at_10.append(1 if rank <= 10 else 0)

        # 替换尾实体
        tail_candidates = torch.arange(num_entities).to(device)
        head_batch = torch.full_like(tail_candidates, h)
        rel_batch = torch.full_like(tail_candidates, r)
        triples = torch.stack([head_batch, rel_batch, tail_candidates], dim=1)

        scores = torch.norm(
            model.entity_embeddings(triples[:, 0]) +
            model.relation_embeddings(triples[:, 1]) -
            model.entity_embeddings(triples[:, 2]),
            p=model.p_norm,
            dim=1
        )

        if filtered:
            for corrupt_t in range(num_entities):
                if corrupt_t != t and (h, r, corrupt_t) in all_triples_set:
                    scores[corrupt_t] = float('inf')

        rank = torch.argsort(scores).tolist().index(t) + 1
        mean_ranks.append(rank)
        hits_at_10.append(1 if rank <= 10 else 0)

    avg_mr = np.mean(mean_ranks)
    avg_hits10 = np.mean(hits_at_10)

    print(f"\nEvaluation Results:")
    print(f"Mean Rank: {avg_mr:.2f}")
    print(f"Hits@10:  {avg_hits10 * 100:.2f}%")

    return avg_mr, avg_hits10
