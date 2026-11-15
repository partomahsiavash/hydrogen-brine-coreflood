import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from run_advanced import ROOT, get_training, make_synthetic
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from demo_config import CHECK_ORIGIN, TARGET
from inverse import WIDTH, Fit, recovery_design
from diagnostics import chain_diagnostics


def stress(config):
    trial, burn, samples = config
    rows=[]
    for tag, frame, gamma, slopes in make_synthetic(noise_seed=39819+trial*163):
        rec=get_training(frame,"RF",CHECK_ORIGIN,.025,1)
        pressure=get_training(frame,"Pinlet",CHECK_ORIGIN,.035,4)
        model=Fit(rec,pressure,True,True)
        posterior,keys=model.sample(burn,samples,35808+trial*317+int(gamma>0))
        link=posterior[...,keys.index("gamma")]
        rf2=100*(posterior[...,:len(WIDTH)]@recovery_design(np.array([TARGET]))[0])
        true_link=60*gamma/pressure.scale
        true_rec=100*(slopes@recovery_design(np.array([TARGET]))[0])
        for key,draws,truth in (("Normalized gamma",link,true_link),("Latent RF at 2 PV (pp)",rf2,true_rec)):
            a,b,c=np.quantile(draws,[.05,.5,.95])
            rhat,ess,tail=chain_diagnostics(draws)
            rows.append((trial,gamma,key,truth,a,b,c,a<=truth<=c,a>0 or c<0,
                         rhat,ess,tail,bool(rhat<1.01 and min(ess,tail)>=400)))
    return rows


def main(args):
    tasks=[(i,args.burn,args.draws) for i in range(1,args.replicates+1)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(stress,tasks):
            rows.extend(result)
    cols=["Replicate","Generating gamma","Quantity","Truth","Q05","Median","Q95","Covered",
          "Excludes zero","Rhat","Bulk ESS","Tail ESS","Diagnostic pass"]
    table=pd.DataFrame(rows,columns=cols)
    out=[]
    for (gamma,key),part in table.groupby(["Generating gamma","Quantity"]):
        n=len(part)
        hits=part.Covered.sum()
        ci=binomtest(int(hits),n).proportion_ci(.95,method="exact")
        out.append((gamma,key,n,hits,hits/n,ci.low,ci.high,part["Excludes zero"].sum() if key=="Normalized gamma" else np.nan,
                    int((~part["Diagnostic pass"]).sum()), np.abs(part.Median-part.Truth).mean()))
    summary=pd.DataFrame(out,columns=["Generating gamma","Quantity","Replicates","Coverage hits","Coverage",
                    "Coverage binomial low","Coverage binomial high","Excludes zero count",
                    "Diagnostics review count","Median prediction MAE"])
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/"synthetic_repeated_trials.csv",index=False)
    summary.to_csv(args.output/"synthetic_repeated_summary.csv",index=False)
    print(summary.to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--replicates",type=int,default=12)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--burn",type=int,default=750)
    parser.add_argument("--draws",type=int,default=1750)
    args=parser.parse_args()
    if min(args.replicates,args.workers,args.burn,args.draws)<=0:
        parser.error("Replicates, workers, burn and draws must be positive.")
    main(args)
