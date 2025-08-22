import math, numpy as np, awkward as ak, uproot, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
#print(torch.__version__)
#print("CUDA available:", torch.cuda.is_available())
def plot_sig_bkg_feature(
    sig, bkg,
    *,
    feature,                 # int column index or str name (if feat_names provided)
    feat_names=None,         # list/tuple mapping names -> column indices
    selected_idx_sig=None,   # per-event chosen pair index (len = B), or None for "all pairs"
    selected_idx_bkg=None,   # same for background
    bins=60,
    xr=None,                 # (xmin, xmax); if None, auto from data
    density=True,
    logy=False,
    ratio=True,
    label_sig="Signal",
    label_bkg="Background",
    xlabel=None,
    ylabel=None,
    vlines=None              # e.g., [125.0] to mark Higgs mass (units = your feature units)
):
    """
    sig, bkg:
      - Either tuples (X, mask) with shapes X:(B,P,D), mask:(B,P) [bool],
        OR 1-D arrays of the feature values.
    feature:
      - If (X,mask) is used: int column index, or str name + feat_names provided.
      - If 1-D arrays are used: ignored.
    selected_idx_*:
      - If provided with (X,mask), pick only the selected pair per event (skip events with idx < 0 or invalid).
      - If None, flattens **all valid pairs**.
    """

    def _to_numpy(x):
        if x is None: return None
        try: import torch
        except Exception: torch = None
        if torch and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _feat_idx_from_name(name, names):
        if names is None:
            raise ValueError("feature is str but feat_names is None.")
        try:
            return list(names).index(name)
        except ValueError:
            raise ValueError(f"feature '{name}' not found in feat_names {names}")

    def _extract(data, feat_idx, selected_idx, w=None):
        """
        Returns:
          vals : 1-D np.array of feature values
          wout : 1-D np.array of weights aligned to vals (if w is provided), else None
        """
        # Case A: already 1-D values
        if not isinstance(data, (tuple, list)):
            vals = _to_numpy(data).astype(float)
            if w is None:
                return vals, None
            w = _to_numpy(w).astype(float)
            if w.shape[0] != vals.shape[0]:
                raise ValueError("weights length must match values length for 1-D input.")
            return vals, w

        # Case B: (X, mask)
        X, mask = data
        X = _to_numpy(X); mask = _to_numpy(mask).astype(bool)
        B, P, D = X.shape

        # Gather values + event indices (so we can broadcast per-event weights)
        vals_list = []
        ev_idx_list = []

        if selected_idx is None:
            for b in range(B):
                vmask = mask[b]
                if not vmask.any(): 
                    continue
                vals_list.append(X[b, vmask, feat_idx])
                ev_idx_list.append(np.full(vmask.sum(), b, dtype=int))
        else:
            sel = _to_numpy(selected_idx).astype(int)
            if sel.shape[0] != B:
                raise ValueError("selected_idx must have length B (number of events).")
            for b in range(B):
                idx = sel[b]
                if idx < 0: 
                    continue
                if idx < P and mask[b, idx]:
                    vals_list.append(np.array([X[b, idx, feat_idx]], dtype=float))
                    ev_idx_list.append(np.array([b], dtype=int))

        if len(vals_list) == 0:
            return np.array([], dtype=float), (None if w is None else np.array([], dtype=float))

        vals = np.concatenate(vals_list, axis=0)
        if w is None:
            return vals, None

        # Build aligned weights:
        w = _to_numpy(w).astype(float)
        if w.shape[0] == vals.shape[0]:
            return vals, w  # already per-value
        if w.shape[0] != B:
            raise ValueError("For (X,mask) input: weights must be length B (per-event) or length Nvals (per-value).")
        ev_idx = np.concatenate(ev_idx_list, axis=0)
        return vals, w[ev_idx]

    # Resolve feature index if needed
    feat_idx = None
    if isinstance(sig, (tuple, list)) or isinstance(bkg, (tuple, list)):
        if isinstance(feature, str):
            feat_idx = _feat_idx_from_name(feature, feat_names)
        else:
            feat_idx = int(feature)

    # Extract arrays + aligned weights
    s_vals, s_w = _extract(sig, feat_idx, selected_idx_sig)
    b_vals, b_w = _extract(bkg, feat_idx, selected_idx_bkg)

    if s_vals.size == 0 or b_vals.size == 0:
        raise ValueError("No values to plot (check masks/selected indices).")

    # Auto-range if needed (robust percentiles)
    if xr is None:
        lo = np.nanmin([np.percentile(s_vals, 1), np.percentile(b_vals, 1)])
        hi = np.nanmax([np.percentile(s_vals, 99), np.percentile(b_vals, 99)])
        xr = (lo, hi)

    # Prepare bins
    bin_edges = np.linspace(xr[0], xr[1], bins+1) if isinstance(bins, int) else np.asarray(bins)

    # Histograms
    h_s, _ = np.histogram(s_vals, bins=bin_edges, range=None, weights=s_w, density=density)
    h_b, _ = np.histogram(b_vals, bins=bin_edges, range=None, weights=b_w, density=density)
    centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # Plot
    if ratio:
        fig = plt.figure(figsize=(6, 5.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        axr = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        axr = None

    # Background
    ax.step(centers, h_b, where="mid", linewidth=1.5, label=label_bkg)
    ax.fill_between(centers, h_b, step="mid", alpha=0.15)

    # Signal
    ax.step(centers, h_s, where="mid", linewidth=1.5, label=label_sig)

    if vlines:
        for v in np.atleast_1d(vlines):
            ax.axvline(float(v), linestyle="--", linewidth=1.0)

    ax.set_xlim(*xr)
    ax.set_yscale("log" if logy else "linear")
    if xlabel is None:
        if isinstance(feature, str):
            xlabel = feature
        elif feat_names is not None and feat_idx is not None:
            xlabel = str(feat_names[feat_idx])
        else:
            xlabel = f"feature[{feat_idx}]" if feat_idx is not None else "value"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ("Density" if density else "Entries"))
    ax.legend()
    ax.grid(alpha=0.25)

    if ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(h_b > 0, h_s / h_b, np.nan)
        axr.step(centers, r, where="mid")
        axr.set_ylabel("S/B")
        axr.set_xlabel(xlabel)
        axr.grid(alpha=0.25)
        axr.set_ylim(bottom=0)  # keep it sane

    fig.tight_layout()
    return fig

def delta_phi( phi1, phi2 ):
    dphi = (phi1-phi2+np.pi)%(2*np.pi) - np.pi 
    return dphi 
def delta_r( eta1, phi1, eta2, phi2):
    return np.sqrt( (eta1-eta2)**2 + delta_phi(phi1,phi2)**2 )
def inv_mass(pt1,eta1,phi1,mass1, pt2,eta2,phi2,mass2):
    def to_cart(pt,eta,phi):
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = pt*np.sinh(eta)
        return px, py, pz
    px1,py1,pz1 = to_cart(pt1,eta1,phi1)
    px2,py2,pz2 = to_cart(pt2,eta2,phi2)
    e1 = np.sqrt(abs(mass1**2 - px1**2 - py1**2 - pz1**2))
    e2 = np.sqrt(abs(mass2**2 - px2**2 - py2**2 - pz2**2))
    m2 = np.maximum( (e1+e2)**2 - ( (px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2), 0.0)
    return np.sqrt(m2)

def featurize_pairs( ev, MH = 125.0):
    nmu = len(ev['mu_pt'])
    idxs=[] 
    feats=[]
    target_pair_index = -1
    true_pair = None 
    if "mu_isFromHiggs" in ev: 
        higgs_mu_idx = np.where(ev["mu_isFromHiggs"])[0]
        if len(higgs_mu_idx)==2: true_pair = tuple(sorted(higgs_mu_idx.tolist()))

    #compute distances to b-jets (if available)
    bj_eta = ev.get("bjet_eta", np.array([]))
    bj_phi = ev.get("bjet_phi", np.array([]))
    njets  = ev.get("n_jets", 0)
    nbjets = ev.get("n_bjets", 0)
    ht     = ev.get("ht", 0.0)
    def min_dr_to_b(eta,phi):
        if bj_eta.size==0: return -1.0
        return np.min(delta_r(eta,phi, bj_eta, bj_phi))

    met = ev.get("met",0.0)
    met_phi= ev.get("met_phi",0.0)
    for i in range(nmu):
        for j in range(i+1, nmu):
            if ev["mu_charge"][i]*ev["mu_charge"][j] >= 0: continue #only OS pairs

            m= inv_mass( ev["mu_pt"][i], ev["mu_eta"][i], ev["mu_phi"][i], ev["mu_m"][i], 
                        ev["mu_pt"][j], ev["mu_eta"][j], ev["mu_phi"][j], ev["mu_m"][j])
            dphi = delta_phi(ev["mu_phi"][i], ev["mu_phi"][j])
            deta = ev["mu_eta"][i] - ev["mu_eta"][j]
            dR   = np.sqrt(dphi**2 + deta**2)
            pt_max = max( ev["mu_pt"][i] , ev["mu_pt"][j] )
            pt_min = min( ev["mu_pt"][i] , ev["mu_pt"][j] )
            pt_ratio = pt_min/pt_max 
            #iso_i = ev.get("mu_iso", np.zeros(nmu))[i]
            dr_b_i = min_dr_to_b( ev["mu_eta"][i], ev["mu_phi"][i])
            dr_b_j = min_dr_to_b( ev["mu_eta"][j], ev["mu_phi"][j])
            pij_x = ev["mu_pt"][i]*np.cos(ev["mu_phi"][i]) + ev["mu_pt"][j]*np.cos(ev["mu_phi"][j])
            pij_y = ev["mu_pt"][i]*np.sin(ev["mu_phi"][i]) + ev["mu_pt"][j]*np.sin(ev["mu_phi"][j])
            pt_ij = np.sqrt( pij_x**2 + pij_y**2)
            phi_ij= math.atan2(pij_y,pij_x)
            dphi_met= delta_phi(phi_ij, met_phi)

            f = [m, abs(m-MH), dR, deta, dphi, pt_max, pt_min, pt_ratio, 
                 dr_b_i, dr_b_j, njets, nbjets, ht, met, dphi_met, pt_ij]
            feats.append(f)
            idxs.append( (i,j))
    #Locate target pair among candidates
    if true_pair is not None: 
        for k, (i,j) in enumerate(idxs):
            if (i,j)== true_pair:
                target_pair_index = k
                break 
    for i, f in enumerate(feats):
        try:
            a = np.asarray(f)
            print(f"[feats[{i}]] shape={a.shape} dtype={a.dtype} value(sample)={a.ravel()[:5]}")
        except Exception as e:
            print(f"[feats[{i}]] type={type(f)} repr={f!r} -> ERR {e}")
    return np.array(feats, dtype=np.float32), idxs, target_pair_index


class EventPairDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ev  = {k: np.array(row[k], dtype=np.float32) if isinstance(row[k], (list, tuple, np.ndarray)) else row[k] for k in row.index}
        feats, idx, target = featurize_pairs(ev)
        return feats, idx, target 

def collate(batch):
    #batch is List of (feats [P_i, d] idxs, target_idx)
    maxP = max(feats.shape[0] for feats,_,_ in batch)
    d    = batch[0][0].shape[1] if maxP>0 else 0
    B    = len(batch)
    X    = torch.zeros(B, maxP, d, dtype=torch.float32 )
    mask = torch.zeros(B, maxP,    dtype= torch.bool)
    y    = torch.full( (B,), -1, dtype=torch.long)
    for b, (feats,idxs,target) in enumerate(batch):
        P = feats.shape[0]
        if P>0:
            X[b,:P] = torch.from_numpy(feats)
            mask[b,:P]= True
        if target>=0:
            y[b]= target 
    return X, mask, y 

class PairRanker(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear( in_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d( 64 ),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,1)
        )
    def forward(self, X,mask):
        B,P,D = X.shape
        Xf    = X.reshape( B*P , D)
        logits= self.net(Xf).reshape(B, P) 
        logits= logits.masked_fill(~mask, -1e9)
        return logits
    
def listwise_ce_loss( logits, targets, mask ):
    #Logits (B,P) , targets: (B,) mask: (B,P)
    # for events without valid target targets[b]<0, ignore loss
    losses= []
    for b in range(logits.shape[0]):
        t = targets[b].item()
        if t>=0 and mask[b].any():
            l = logits[b]
            l = l - torch.logsumexp(l[mask[b]], dim=0) #Log softmax over valid pairs
            losses.append( -l[t])
    if len(losses) ==0:
        return torch.tensor(0.0, requires_grad = True)
    return torch.stack(losses).mean()

def train(parq_train, parq_val, epochs=10, lr=1e-3, batch_size=128):
    from metrics_helper import MetricsTracker

    tracker = MetricsTracker() # tracker of metrics

    ds_tr = EventPairDataset(parq_train)
    ds_val= EventPairDataset(parq_val)
    dl_tr = DataLoader(ds_tr , batch_size=batch_size, shuffle=True , collate_fn= collate)
    dl_va = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn= collate)
    print(dl_tr)
    #Feature dims
    X0, m0, _ = next(iter(dl_tr))
    in_dim    = X0.shape[-1]
    model     = PairRanker( in_dim )
    opt       = torch.optim.AdamW( model.parameters(), lr= lr)

    best_val = float('inf')
    for ep in range(1, epochs+1):
        #Training
        model.train()
        tr_loss = 0
        nb = 0
        for X, mask, y in dl_tr:
            #print("train X : ",X)
            #print("train mask : ",mask)
            #print("train y : ",y)
            opt.zero_grad()
            logits = model(X,mask)
            loss = listwise_ce_loss(logits, y, mask )
            loss.backward()
            opt.step()
            tr_loss+= loss.item()
            nb+=1
        tr_loss /= max(nb,1)
        #Validatioj
        model.eval()
        va_loss = 0
        vb=0 
        tracker.reset_epoch()
        acc_n = 0
        acc_d =0
        with torch.no_grad():
            for X,mask,y in dl_va:
                logits = model(X, mask)
                loss   = listwise_ce_loss(logits, y, mask )
                va_loss += loss.item()
                tracker.update_batch(logits, mask, y, X=X, mass_col=0)
                vb +=1 
                for b in range(X.shape[0]):
                    if y[b].item()>=0 and mask[b].any():
                        valid = torch.where(mask[b])[0]
                        pred  = torch.argmax( logits[b,valid]).item()
                        if pred == y[b].item(): acc_n +=1
                        acc_d +=1
        va_loss /= max(vb,1)
        metrics = tracker.compute_epoch_metrics(topk=(1,2,3))
        tracker.log_epoch(epoch=ep, train_loss=tr_loss, val_loss=va_loss, **metrics)
        print(f"[ep {ep}] train {tr_loss:.4f} | val {va_loss:.4f} | accur {metrics['top1_acc']:.3f} | AUC {metrics['roc_auc']:.3f}")
        
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model" : model.state_dict(), "in_dim": in_dim}, "pair_ranker.pt")
    
    tracker.save_history('metrics')
    tracker.plot_learning_curves('metrics')
    tracker.plot_rank_histogram('metrics')
    tracker.plot_roc_pr_curves('metrics')

if __name__ == "__main__":
    import argparse
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=str, default="train_pairs.parquet")
    ap.add_argument('--val',  type=str, default="valid_pairs.parquet")
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    print(args)
    validDS_path =os.path.join(script_dir, args.val)
    trainDS_path =os.path.join(script_dir, args.train)
    train(trainDS_path, validDS_path, epochs=args.epochs, lr=args.lr, batch_size=args.batch)