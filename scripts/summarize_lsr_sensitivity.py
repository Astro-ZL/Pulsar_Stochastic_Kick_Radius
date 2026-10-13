"""Compare the main result and recorded Lorimer-prior medians."""
from pathlib import Path
import csv,pickle
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 with (ROOT/'results/lsr_sensitivity/lorimer2006_recorded_medians.csv').open() as f:rows=list(csv.DictReader(f))
 d=pickle.load((ROOT/'results/mcmc/posterior_all_77_11param.pkl').open('rb'))
 out=[]
 for r in rows:
  c=r['parameter'];q=np.percentile(d['samples'][:,d['columns'].index(c)],[16,50,84]);v=float(r['median'])
  out.append(dict(parameter=c,main_q16=q[0],main_median=q[1],main_q84=q[2],lorimer_recorded_median=v,median_difference=v-q[1],median_relative_change_percent=100*(v-q[1])/q[1]))
 dest=ROOT/'reproduced/summaries/median_comparison_77_LN_11.csv';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 for r in out:print(r['parameter'],f"main={r['main_median']:.4f}; Lorimer={r['lorimer_recorded_median']:.4f}; change={r['median_relative_change_percent']:.3f}%")
 print('Saved',dest)
if __name__=='__main__':main()
