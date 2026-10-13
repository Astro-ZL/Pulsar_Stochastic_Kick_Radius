"""Summarize frozen files without refitting or modifying them."""
from pathlib import Path
import csv, json, pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def model_rows():
    config=json.loads((ROOT/'config.json').read_text())
    rows=[]
    for model in config['mcmc_models']:
        d=pickle.load((ROOT/model['posterior']).open('rb'))
        for i,c in enumerate(d['columns']):
            q=np.percentile(d['samples'][:,i],[16,50,84])
            rows.append(dict(sample=model['sample'],velocity_model=model['velocity_model'],structure=model['structure'],parameter=c,q16=q[0],median=q[1],q84=q[2],err_minus=q[1]-q[0],err_plus=q[2]-q[1],n_samples=len(d['samples'])))
    return rows
def evidence_rows():
    config=json.loads((ROOT/'config.json').read_text());rows=[]
    for model in config['evidence_models']:
        folder=ROOT/model['results']
        a=pickle.load((folder/'posterior_equal_full.pkl').open('rb'))
        b=pickle.load((folder/'posterior_equal_zero.pkl').open('rb'))
        delta=float(a['logz']-b['logz']);err=float(np.hypot(a['logzerr'],b['logzerr']))
        rows.append(dict(model=model['label'],logZ_full=float(a['logz']),logZ_null=float(b['logz']),lnBF=delta,lnBF_error=err,BF=float(np.exp(delta)),n_sim=a['n_sim'],nlive_full=a['nlive'],nlive_null=b['nlive'],dlogz=a['dlogz']))
    return rows
def save_rows(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':
    save_rows(model_rows(),ROOT/'reproduced/summaries/model_parameter_summary.csv')
    save_rows(evidence_rows(),ROOT/'reproduced/summaries/bayes_factor_summary.csv')
    print('Summaries saved to reproduced/summaries; frozen results are unchanged.')
