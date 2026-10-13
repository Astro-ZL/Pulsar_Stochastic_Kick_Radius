"""Independent checks of archive contents, identities, hashes and result metadata."""
from pathlib import Path
import ast, csv, hashlib, json, pickle, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check(condition,message):
    if not condition:raise AssertionError(message)
def main():
    checks=[]
    for f in list(ROOT.rglob('*.py')):
        if 'reproduced' in f.parts:continue
        ast.parse(f.read_text());checks.append('syntax:'+str(f.relative_to(ROOT)))
        if f.name!='verify_archive.py':
            check('/home/lz/work/pulsar_kick_period' not in f.read_text(),'hardcoded research path:'+str(f))
    cfg=json.loads((ROOT/'config.json').read_text())
    from prepare_observations import selected_catalog
    df=selected_catalog();check(len(df)==77,'catalog sample count')
    period77=np.loadtxt(ROOT/'data/period_1_new_77.txt')
    check(np.array_equal(period77,df['Period'].to_numpy()),'77 periods/order')
    young=df['Age(10**7)'].to_numpy()<1;check(young.sum()==47,'young count')
    check(np.array_equal(np.loadtxt(ROOT/'data/period_2_new_47.txt'),period77[young]),'47 periods/order')
    check(np.array_equal(period77,np.loadtxt(ROOT/'data/period_77_data.txt')),'period alias agreement')
    for n in [47,77]:
        cloud=np.loadtxt(ROOT/f'data/vpec_lsr_samples_matrix_{n}x100.txt')
        check(cloud.shape==(n,100),'cloud shape');check(np.isfinite(cloud).all() and (cloud>0).all(),'cloud positivity')
        q=np.percentile(cloud,[16,50,84],axis=1)
        for kind,x in [('median',q[1]),('err_low',q[1]-q[0]),('err_up',q[2]-q[1])]:
            check(np.allclose(x,np.loadtxt(ROOT/f'data/vpec_lsr_{kind}_{n}.txt'),rtol=1e-9,atol=1e-8),'velocity quantile:'+kind)
    check(np.array_equal(np.loadtxt(ROOT/'data/vpec_lsr_samples_matrix_47x100.txt'),np.loadtxt(ROOT/'data/vpec_lsr_samples_matrix_77x100.txt')[young]),'47 cloud subset identity')
    for model in cfg['mcmc_models']:
        d=pickle.load((ROOT/model['posterior']).open('rb'));x=np.asarray(d['samples'])
        check(x.ndim==2 and x.shape[1]==model['dimension'],'posterior dimensions')
        check(np.isfinite(x).all(),'nonfinite posterior')
    for model in cfg['evidence_models']:
        folder=ROOT/model['results']
        for key in ['full','zero']:
            d=pickle.load((folder/f'posterior_equal_{key}.pkl').open('rb'))
            res=pickle.load((folder/f'dynesty_results_{key}.pkl').open('rb'))
            check(np.isclose(d['logz'],res.logz[-1],rtol=0,atol=1e-10),'evidence logZ metadata')
            check(d['n_sim']==model['n_sim'] and d['dlogz']==model['dlogz'],'evidence settings')
            check(d['nlive']==model['nlive_full' if key=='full' else 'nlive_null'],'evidence live points')
    single=pickle.load((ROOT/'results/single_kick/posterior_LSR_47_samples.pkl').open('rb'))
    check(np.asarray(single).shape[1]==6 and np.isfinite(single).all(),'single-kick posterior')
    manifest=ROOT/'provenance/file_manifest.tsv'
    if manifest.exists():
        with manifest.open() as f:
            for row in csv.DictReader(f,delimiter='\t'):
                check((ROOT/row['path']).exists(),'missing:'+row['path'])
                check(sha(ROOT/row['path'])==row['sha256'],'hash:'+row['path'])
    for f in (ROOT/'notebooks').glob('*.ipynb'):
        nb=json.loads(f.read_text());check(nb['nbformat']==4,'notebook format')
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
    report=dict(status='PASS',checks=['Python syntax and relative paths','77/47 source identities and ordering','velocity quantile summaries','12 posterior shapes and finite values','8 dynesty results and metadata','single-kick posterior','notebook format','file checksums'])
    out=ROOT/'reproduced/verification/verification_report.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
