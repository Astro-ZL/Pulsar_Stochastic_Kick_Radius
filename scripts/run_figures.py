"""Regenerate figures from archived samples; no inference runs."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
FILES=['plot_radius_comparison.py','plot_maxwell_corner.py','plot_posterior_predictive.py','plot_lsr_comparison.py']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['all','1','2','3','A1'],default='all');args=parser.parse_args()
    env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    chosen=FILES if args.figure=='all' else [dict(zip(['1','2','3','A1'],FILES))[args.figure]]
    for name in chosen:
        print('Generating',name,flush=True);subprocess.run([sys.executable,str(ROOT/'Figures'/name)],cwd=ROOT,env=env,check=True)
    print('Regenerated files are in reproduced/figures.')
