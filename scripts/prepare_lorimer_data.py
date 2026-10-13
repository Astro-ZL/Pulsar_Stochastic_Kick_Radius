"""Generate distances and LSR velocities with the Lorimer Model C prior."""
from pathlib import Path
import argparse,json
import numpy as np
from prepare_observations import selected_catalog
from lorimer_distance_velocity import generate_lsr_velocity_matrix_for_sample_lorimer
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--execute-preprocess',action='store_true')
 p.add_argument('--n-mc',type=int,default=10000)
 p.add_argument('--output',type=Path,default=ROOT/'reproduced/data/lorimer2006')
 args=p.parse_args()
 if not args.execute_preprocess:p.error('Use --execute-preprocess to generate samples; supplied data are in data/lorimer2006/.')
 if args.n_mc<100:p.error('n-mc must be at least 100')
 df=selected_catalog()
 a,b,c,summary=generate_lsr_velocity_matrix_for_sample_lorimer(df['Jname'],df['gl'],df['gb'],df['pmra'],df['pmdec'],df['Dist'],df['Dist_eminus'],df['Dist_eplus'],n_mc=args.n_mc,seed0=20260610)
 args.output.mkdir(parents=True,exist_ok=True)
 for kind,x in [('distance',a),('vtrans_helio',b),('vpec_lsr',c)]:
  np.savetxt(args.output/f'{kind}_lorimerC_samples_matrix_77x{args.n_mc}.txt',x,fmt='%.10e')
  compressed=np.quantile(x,(np.arange(100)+0.5)/100,axis=1).T
  np.savetxt(args.output/f'{kind}_lorimerC_quantile_cloud_77x100.txt',compressed,fmt='%.10e')
  q=np.percentile(x,[16,50,84],axis=1)
  for suffix,v in [('median',q[1]),('err_low',q[1]-q[0]),('err_up',q[2]-q[1])]:np.savetxt(args.output/f'{kind}_lorimerC_{suffix}_77.txt',v,fmt='%.10e')
 (args.output/'source_summaries.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('Saved samples to',args.output)
if __name__=='__main__':main()
