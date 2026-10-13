"""Check the Lorimer sample identities, quantile clouds and recorded comparison."""
from pathlib import Path
import ast,csv,json,pickle
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def verify():
 folder=ROOT/'data/lorimer2006'
 df=pd.read_csv(ROOT/'data/catalog_selected_77.csv')
 summary=pd.read_csv(folder/'lorimerC_distance_velocity_summary_77.csv')
 assert len(df)==len(summary)==77
 assert np.array_equal(df['Jname'].str.strip(),summary['source_name'].str.strip())
 for kind in ['distance','vtrans_helio','vpec_lsr']:
  x=np.loadtxt(folder/f'{kind}_lorimerC_samples_matrix_77x10000.txt')
  assert x.shape==(77,10000) and np.isfinite(x).all() and (x>0).all()
  compressed=np.loadtxt(folder/f'{kind}_lorimerC_quantile_cloud_77x100.txt')
  q=np.quantile(x,(np.arange(100)+0.5)/100,axis=1).T
  assert compressed.shape==(77,100) and np.allclose(q,compressed,rtol=1e-9,atol=1e-8)
  p=np.percentile(x,[16,50,84],axis=1)
  for suffix,v in [('median',p[1]),('err_low',p[1]-p[0]),('err_up',p[2]-p[1])]:
   assert np.allclose(v,np.loadtxt(folder/f'{kind}_lorimerC_{suffix}_77.txt'),rtol=1e-9,atol=1e-8)
 columns={'distance':('d_median_kpc','d_err_minus_kpc','d_err_plus_kpc'),
          'vtrans_helio':('Vtrans_helio_median_km_s','Vtrans_helio_err_minus_km_s','Vtrans_helio_err_plus_km_s'),
          'vpec_lsr':('Vpec_LSR_median_km_s','Vpec_LSR_err_minus_km_s','Vpec_LSR_err_plus_km_s')}
 for kind,cs in columns.items():
  for c,suffix in zip(cs,['median','err_low','err_up']):
   assert np.allclose(summary[c],np.loadtxt(folder/f'{kind}_lorimerC_{suffix}_77.txt'),rtol=1e-9,atol=1e-8)
 main=pickle.load((ROOT/'results/mcmc/posterior_all_77_11param.pkl').open('rb'))
 with (ROOT/'results/lsr_sensitivity/lorimer2006_recorded_medians.csv').open() as f:record=list(csv.DictReader(f))
 comparison=pd.read_csv(ROOT/'results/lsr_sensitivity/median_comparison_77_LN_11.csv')
 assert len(record)==len(comparison)==11
 output=(ROOT/'results/lsr_sensitivity/lorimer2006_mcmc_output.txt').read_text()
 assert '100.0% (2000/2000)' in output
 logged=[float(line.rsplit(':',1)[1]) for line in output.splitlines() if line.startswith('$') and ':' in line]
 assert np.array_equal(logged,[float(r['median']) for r in record])
 for i,r in enumerate(record):
  c=r['parameter'];q=np.percentile(main['samples'][:,main['columns'].index(c)],[16,50,84]);v=float(r['median'])
  row=comparison.iloc[i]
  assert row['parameter']==c and np.allclose(q,row[['main_q16','main_median','main_q84']].to_numpy(dtype=float),rtol=0,atol=1e-12)
  assert np.isclose(row['lorimer_recorded_median'],v,rtol=0,atol=1e-12)
  assert np.isclose(row['median_difference'],v-q[1],rtol=0,atol=1e-12)
 def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
 a=funcs(ROOT/'src/mcmc/run_LSR_77_LN_11.py');b=funcs(ROOT/'src/lsr_sensitivity/run_lorimer2006_77_LN_11.py')
 for n in ['simulate_full_population','log_prior','log_likelihood','run_single_chain']:assert a[n]==b[n]
 print('PASS: Lorimer source ordering, 3 full sample matrices, 3 quantile clouds, summaries, recorded medians and shared model functions.')
if __name__=='__main__':verify()
