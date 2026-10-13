"""Recreate the final parallax-based LSR inputs from the supplied catalog."""
from pathlib import Path
import argparse, csv
import numpy as np
import pandas as pd
from lsr_velocity import generate_lsr_velocity_matrix_for_sample
ROOT=Path(__file__).resolve().parents[1]
def selected_catalog():
    df=pd.read_excel(ROOT/'data/parallaxes_1.xlsx')
    # These column names contain parallax, in mas, despite 'Dist'.
    mask=(df['Dist']/df['Dist_eminus']>3)&(df['Jname']!='J1809-1943')&(df['Period']>0.1)
    return df.loc[mask].reset_index(drop=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reproduced/data')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    df=selected_catalog();assert len(df)==77
    matrix,summary=generate_lsr_velocity_matrix_for_sample(df['Jname'],df['gl'],df['gb'],df['pmra'],df['pmdec'],df['Dist'],df['Dist_eminus'],df['Dist_eplus'],n_mc=100,seed0=20260610)
    young=df['Age(10**7)'].to_numpy()<1
    for n,selected in [(77,np.ones(77,dtype=bool)),(47,young)]:
        assert selected.sum()==n
        cloud=matrix[selected];q=np.percentile(cloud,[16,50,84],axis=1)
        np.savetxt(args.output/f'vpec_lsr_samples_matrix_{n}x100.txt',cloud,fmt='%.10e')
        for kind,values in [('median',q[1]),('err_low',q[1]-q[0]),('err_up',q[2]-q[1])]:
            np.savetxt(args.output/f'vpec_lsr_{kind}_{n}.txt',values,fmt='%.10e')
        name='period_1_new_77.txt' if n==77 else 'period_2_new_47.txt'
        np.savetxt(args.output/name,df['Period'].to_numpy()[selected],fmt='%.5e')
    print('Recreated data saved to',args.output)
if __name__=='__main__':main()
