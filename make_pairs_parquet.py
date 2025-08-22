
import uproot, awkward as ak, numpy as np, pandas as pd
from pathlib import Path
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(script_dir, "ttH.root")
#IN = "ttH.root"
OUT_TR  =  os.path.join(script_dir, "train_pairs.parquet")
OUT_VAL =  os.path.join(script_dir, "valid_pairs.parquet")
SPLIT   = 0.8

MU = {
  'pt': 'MuonsVector_pt', 
  'eta': 'MuonsVector_eta', 
  'phi': 'MuonsVector_phi', 
  'm'  : 'MuonsVector_m',
  'q': 'MuonsVector_charge',
  #'iso': 'Muons_Pos_IsoPassTight', 
  #'d0sig': 'Muon_d0sig', 
  #'z0s': 'Muon_z0sintheta',
  'fromH': 'MuonsVector_truthOrigin'  # <-- boolean per muon that YOU must create upstream
}

JET = {'eta': 'Jets_Eta', 
       'phi': 'Jets_Phi', 
       'btag': 'Jets_LowestPassedBTagOP'}
EVT = {'met': 'Event_MET', 
       'metphi': 'Event_MET_Phi', 
       'nj': 'Jets_jetMultip', 
       #'nb': 'nBTag', 
       'ht':'Event_Ht'}

with uproot.open(IN) as f:
    t = f['tree_Hmumu']
    arrs = t.arrays(list(MU.values()) + list(JET.values()) + list(EVT.values()),
                    library="ak", how=dict)
#how='ak')

# Build per-event dicts
records = []
for i in range(len(arrs)):  # event loop (vectorize later if needed)
    rec = {
      'mu_pt': np.array(arrs[MU['pt']][i]),
      'mu_eta': np.array(arrs[MU['eta']][i]),
      'mu_phi': np.array(arrs[MU['phi']][i]),
      'mu_m'  : np.array(arrs[MU['m']][i]),
      'mu_charge': np.array(arrs[MU['q']][i]),
      #'mu_iso': np.array(arrs[MU['iso']][i]) if MU['iso'] in arrs.fields else [],
      #'mu_d0sig': np.array(arrs[MU['d0sig']][i]) if MU['d0sig'] in arrs.fields else [],
      #'mu_z0sintheta': np.array(arrs[MU['z0s']][i]) if MU['z0s'] in arrs.fields else [],
      'mu_isFromHiggs': np.array(arrs[MU['fromH']][i], dtype=bool),
      'bjet_eta': np.array(arrs[JET['eta']][i][arrs[JET['btag']][i] > 0.5]) if JET['btag'] in arrs else np.array([]),
      'bjet_phi': np.array(arrs[JET['phi']][i][arrs[JET['btag']][i] > 0.5]) if JET['btag'] in arrs else np.array([]),
      'n_jets': int(arrs[EVT['nj']][i]) if EVT['nj'] in arrs else 0,
      #'n_btags': int(arrs[EVT['nb']][i]) if EVT['nb'] in arrs.fields else 0,
      'ht': float(arrs[EVT['ht']][i]) if EVT['ht'] in arrs else 0.0,
      'met': float(arrs[EVT['met']][i]) if EVT['met'] in arrs else 0.0,
      'met_phi': float(arrs[EVT['metphi']][i]) if EVT['metphi'] in arrs else 0.0,
    }
    if len(rec['mu_pt']) >= 2:
        records.append(rec)

# Split & save
n = len(records)
k = int(SPLIT*n)
tr = pd.DataFrame(records[:k])
va = pd.DataFrame(records[k:])
tr.to_parquet(OUT_TR)
va.to_parquet(OUT_VAL)
print('Wrote', OUT_TR, OUT_VAL)