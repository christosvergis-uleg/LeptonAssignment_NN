import os, json, math, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

class MetricsTracker():
    """
    Tracks epoch history and computes validation metrics for the PairRanker.

    Usage in validation loop per epoch:
        tracker.reset_epoch()
        for X, mask, y in dl_val:
            logits = model(X, mask)         # (B, Pmax)
            tracker.update_batch(logits, mask, y, X=X, mass_col=0)  # X optional; use if you want mass plots
        metrics = tracker.compute_epoch_metrics(topk=(1,2,3))
        tracker.log_epoch(train_loss=..., val_loss=..., **metrics)

    After training:
        tracker.save_history(out_dir)
        tracker.plot_learning_curves(out_dir)
        tracker.plot_rank_histogram(out_dir)
        tracker.plot_roc_pr_curves(out_dir)   # if sklearn available
    """
    def __init__(self):
        self.history={
            'epoch' : [],
            'train_loss': [], 'val_loss':[],
            'top1_acc':[], 'top2_acc':[], 'top3_acc':[],
            'roc_auc':[], 'pr_auc':[],
        }
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._correct = 0  
        self._count = 0    
        self._true_ranks = [] # 1 = best rank
        self._all_pair_scores = [] # for ROC/PR
        self._all_pair_labels= [] # 1 for true pair else 0
        self._mass_selected  = [] #selected pair mass 
        self._mass_all_pairs = [] #list of arrays per event


    def reset_epoch(self):
        self._reset_accumulators()

    @staticmethod
    def _masked_argmax(v,m):
        # v: (Pmax,) , m: (Pmax,) bool
        if not np.any(m): return -1
        idx = np.argmax(np.where(m,v,-np.inf))
        return int(idx)
    
    def update_batch(self, logits, mask, targets, X=None, mass_col=None):
        """
        Accumulate metrics from a batch.
        logits: (B,Pmax) torch.Tensor
        mask:   (B,Pmax) torch.BoolTensor
        targets:(B,)     torch.LongTensor  (−1 means ignore)
        X:      (B,Pmax,D) torch.Tensor (optional, for extracting masses)
        mass_col: int or None (column index of mμμ in X)
        """
        import torch 
        B, Pmax = logits.shape
        l_np    = logits.detach().cpu().numpy()
        m_np    = mask.detach().cpu().numpy()
        t_np    = targets.detach().cpu().numpy()
        masses  = None
        if X is not None and mass_col is not None:
            masses = X[:,:,mass_col].detach().cpu().numpy() # (B,Pmax)

        for b in range(B):
            valid = m_np[b].astype(bool)
            if not valid.any(): continue 

            pred = self._masked_argmax(l_np[b], valid)
            if t_np[b] >=0:
                self._count+=1
                if pred == t_np[b]:
                    self._correct +=1
                order = np.argsort(-l_np[b][valid])  # indices within valid subset
                valid_idx = np.where(valid)[0]
                true_local=  np.where(order == np.where(valid_idx == t_np[b])[0][0])[0][0]
                self._true_ranks.append(int(true_local)+1)
                #pair lvl labels/scores for ROC/PR
                labels = np.zeros(valid.sum(), dtype=np.int64)
                #mark the true pair within the valid subset
                true_pos_local = np.where(valid_idx == t_np[b])[0][0]
                labels[true_pos_local] = 1
                scores  = l_np[b][valid]
                self._all_pair_labels.append(labels)
                self._all_pair_scores.append(scores)
            
            if masses is not None:
                self._mass_all_pairs.append(masses[b][valid])
                if pred>=0:
                    self._mass_selected.append(masses[b][pred])
                
    def compute_epoch_metrics(self, topk=(1,2,3)):
        #top-k accuracies from rank histogram
        topk_acc = {}
        ranks    = np.array(self._true_ranks, dtype=np.int64)
        for k in topk:
            if len(ranks)>0 : topk_acc[f'top{k}_acc']= float(np.mean(ranks<=k))
            else:             topk_acc[f'top{k}_acc']=float('nan')
        #ROC/PR AUC on pair-level
        y = np.concatenate(self._all_pair_labels)
        s = np.concatenate(self._all_pair_scores)
        if y.max() > y.min():
            roc = float(roc_auc_score(y,s))
            pr  = float(average_precision_score(y,s))
        else:
            roc = float('nan')
            pr  = float('nan')
        
        return {
            'top1_acc': topk_acc.get('top1_acc', float('nan')),
            'top2_acc': topk_acc.get('top2_acc', float('nan')),
            'top3_acc': topk_acc.get('top3_acc', float('nan')),
            'roc_auc': roc,
            'pr_auc': pr,
        }

    def log_epoch(self, epoch=None, train_loss=None, val_loss=None, **metrics):
        if epoch is None: epoch = len(self.history['epoch']) + 1
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(float(train_loss) if train_loss is not None else None)
        self.history['val_loss'].append(float(val_loss) if val_loss is not None else None)
        for k in ['top1_acc','top2_acc','top3_acc','roc_auc','pr_auc']:
            self.history[k].append(metrics.get(k, None))
    
    def save_history(self, out_dir = 'metrics'):
        os.makedirs(out_dir, exist_ok = True)
        with open(os.path.join(out_dir, 'history.json'),'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_learning_curves(self, out_dir='metrics'):
        os.makedirs(out_dir, exist_ok=True)
        ep = self.history['epoch']
        # Losses
        plt.figure(figsize=(5,4))
        plt.plot(ep, self.history['train_loss'], label='train')
        plt.plot(ep, self.history['val_loss'], label='validation')
        plt.xlabel('epoch') 
        plt.ylabel('Loss') 
        plt.title('Loss vs epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss_vs_epoch.png'), dpi=160)
        plt.close()
        # Accuracy
        plt.figure(figsize=(5,4))
        plt.plot(ep, self.history['top1_acc'], label='top1')
        plt.plot(ep, self.history['top2_acc'], label='top2')
        plt.plot(ep, self.history['top3_acc'], label='top3')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Top‑k accuracy vs epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'topk_vs_epoch.png'), dpi=160)
        plt.close()
        # AUCs (if avail)
        if any(x is not None for x in self.history['roc_auc']):
            plt.figure(figsize=(5,4))
            plt.plot(ep, self.history['roc_auc'], label='ROC AUC')
            plt.plot(ep, self.history['pr_auc'], label='PR AUC')
            plt.xlabel('epoch')
            plt.ylabel('AUC')
            plt.title('Pair‑level AUCs vs epoch')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'auc_vs_epoch.png'), dpi=160)
            plt.close()
    
    def plot_rank_histogram( self, out_dir='metrics', max_rank=10):
        os.makedirs(out_dir, exist_ok=True)
        ranks = np.array(self._true_ranks, dtype=np.int64)
        if ranks.size == 0: return 
        bins = np.arange(1, max_rank+2)
        
        hist, _ = np.histogram(np.clip(ranks, 1, max_rank), bins=bins)
        plt.figure(figsize=(5,4))
        plt.bar(np.arange(1, max_rank+1), hist)
        plt.xlabel('True pair rank (1=best)')
        plt.ylabel('count')
        plt.title('Where does the true pair rank?')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'true_rank_hist.png'), dpi=160)
        plt.close()
    
    def plot_roc_pr_curves(self, out_dir='metrics'):
        if len(self._all_pair_labels) == 0:
            return
        from sklearn.metrics import roc_curve, precision_recall_curve
        y = np.concatenate(self._all_pair_labels)
        s = np.concatenate(self._all_pair_scores)
        # Guard: need both classes
        if y.max() == y.min():
            return
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Pair‑level ROC')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=160)
        plt.close()
        plt.figure(figsize=(5,4))
        plt.plot(rec, prec)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Pair‑level PR')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=160)
        plt.close()

    def pair_confusion_matrix(self, threshold=0.5):
        """
        Optional: treat each candidate pair as a binary example (true pair = positive),
        and threshold the *eventwise* probability to mark predictions.
        Note: because probabilities sum to 1 per event, set threshold accordingly (e.g., 1/P).
        Returns (cm, labels) where cm is 2×2 [[TN, FP],[FN, TP]].
        """
        if len(self._all_pair_labels) == 0:
            return None, ['neg','pos']
        y = np.concatenate(self._all_pair_labels)
        s = np.concatenate(self._all_pair_scores)
        # Convert logits to per-pair probabilities via softmax across each event
        # Here we approximate by applying a global sigmoid-like threshold; better to store per-event softmax.
        # For sanity, just use a percentile-based threshold if given 0.0< threshold <1.0
        if threshold <= 0 or threshold >= 1:
            thr = np.quantile(s, 0.9)  # fallback
        else:
            thr = threshold
        y_pred = (s >= thr).astype(int)
        
        cm = confusion_matrix(y, y_pred)
        return cm, ['neg','pos']
    