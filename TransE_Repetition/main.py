import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transe_model import TransE
from transe_dataset import TransEDataset
from evaluate import evaluate_transe

# 加载三元组
def load_triples_id_format(file_path, max_samples=None):
    triples = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if max_samples is not None:
            lines = lines[:max_samples]
        if lines[0].strip().isdigit():
            lines = lines[1:]
        for line in lines:
            h, t, r = map(int, line.strip().split())
            triples.append((h, r, t))
    return triples

# 训练TransE模型
def train_transe(train_data, num_entities, num_relations, device, save_path,
                 embedding_dim=100, margin=1.0, learning_rate=0.001, num_epochs=10):
    model = TransE(num_entities, num_relations, embedding_dim=embedding_dim, margin=margin, p_norm=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TransEDataset(train_data, num_entities)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    print("Start training...")
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for pos_batch, neg_batch in dataloader:
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)

            pos_score, neg_score = model(pos_batch, neg_batch)
            loss = model.loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.entity_embeddings.weight.data = F.normalize(model.entity_embeddings.weight.data, p=2, dim=1)
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    return model

def main():
    # --- 自定义超参数入口 ---
    embedding_dim = 100       # 嵌入维度
    margin = 1.0              # 间隔超参数
    learning_rate = 0.001     # 学习率
    num_epochs = 20           # 训练轮数
    max_train_samples = 5000  # 只取前n个训练样本

    # --- 加载数据 ---
    data_dir = "./data"
    train_triples = load_triples_id_format(os.path.join(data_dir, 'train2id.txt'), max_samples=max_train_samples)
    valid_triples = load_triples_id_format(os.path.join(data_dir, 'valid2id.txt'))
    test_triples  = load_triples_id_format(os.path.join(data_dir, 'test2id.txt'))

    # --- 统计实体和关系数量 ---
    all_entities = set()
    all_relations = set()
    for h, r, t in train_triples + valid_triples + test_triples:
        all_entities.update([h, t])
        all_relations.add(r)

    num_entities = max(all_entities) + 1
    num_relations = max(all_relations) + 1

    print("Number of entities:", num_entities)
    print("Number of relations:", num_relations)
    print("Example triple:", train_triples[0])

    # --- 确定设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = "transe_model.pt"

    # --- 训练模型 ---
    model = train_transe(
        train_data=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        device=device,
        save_path=save_path,
        embedding_dim=embedding_dim,
        margin=margin,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    # --- 评估模型 ---
    all_triples_set = set(train_triples + valid_triples + test_triples)

    evaluate_transe(
        model=model,
        data=test_triples,
        all_triples_set=all_triples_set,
        num_entities=num_entities,
        device=device,
        filtered=True
    )

if __name__ == "__main__":
    main()
