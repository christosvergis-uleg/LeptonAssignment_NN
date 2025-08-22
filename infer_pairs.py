import torch, pandas as pd, numpy as np
from train_pair_ranker import PairRanker, collate, featurize_pairs

ckpt = torch.load('pair_ranker.pt', map_location='cpu')
model = PairRanker(ckpt['in_dim']); model.load_state_dict(ckpt['model']); model.eval()

val = pd.read_parquet('val_pairs.parquet')

def pick_pair(ev):
    feats, idxs, _ = featurize_pairs(ev)
    X = torch.from_numpy(feats).unsqueeze(0)  # (1,P,D)
    mask = torch.ones(1, X.shape[1], dtype=torch.bool)
    with torch.no_grad():
        logits = model(X, mask)
    best = int(torch.argmax(logits[0]).item())
    return idxs[best], logits[0].softmax(dim=0)[best].item()

for i in range(len(val)):
    ev = {k: np.array(val.iloc[i][k]) if isinstance(val.iloc[i][k], list) else val.iloc[i][k] for k in val.columns}
    (i_mu, j_mu), conf = pick_pair(ev)
    print(i, i_mu, j_mu, conf)