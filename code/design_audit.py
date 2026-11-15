import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inverse import EDGES, WIDTH, recovery_design
from demo_config import PRIMARY, TEST, TARGET
from result_tables import read_result


def run(root):
    diagnostics = []
    manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
    d=read_result(root,"source_observations")
    for c in d.Condition.unique():
        for cutoff in manifest["origins"]:
            f=d[(d.Condition==c)&(d.PV<=cutoff+1e-10)]
            rx=f[f.Channel=="RF"].PV.to_numpy()
            px=f[f.Channel=="Pinlet"].PV.to_numpy()
            rf_mat, pin_mat=recovery_design(rx), recovery_design(px)
            for i,(a,b) in enumerate(zip(EDGES[:-1],EDGES[1:])):
                diagnostics.append((c,cutoff,i+1,a,b,not np.any(rf_mat[:,i]),
                                    not np.any(np.r_[rf_mat[:,i],pin_mat[:,i]]),
                                    max(0.,min(b,TARGET)-a)))
    cols=["Condition","Origin","Rate","PV start","PV end","RF likelihood blind",
          "Joint mean likelihood blind","Contribution weight to RF at 2 PV"]
    pd.DataFrame(diagnostics,columns=cols).to_csv(root/"design_blind_intervals.csv",index=False)


    cand=np.linspace(PRIMARY+.05, TEST, 16).round(4)
    targets=(TARGET, 2.6, EDGES[-1])
    design=recovery_design(np.r_[cand,targets])
    gains=[]
    with np.load(root/"posterior_chains.npz") as posterior:
        for fit in manifest["posterior_arrays"]:
            conf=fit["scenario"]
            if fit["condition"].startswith("Synthetic") or not conf["name"].startswith("RF "):
                continue
            draws=posterior[fit["array"]][...,:len(WIDTH)].reshape(-1,len(WIDTH))
            z=100*(draws@design.T)
            cv=np.cov(z,rowvar=False)
            for noise in (1.,2.):
                for j,x in enumerate(cand):
                    for k,target in enumerate(targets,len(cand)):
                        pct=100*cv[j,k]**2/(cv[k,k]*(cv[j,j]+noise**2))
                        gains.append((fit["condition"],conf["name"],x,target,noise,
                                      np.sqrt(cv[k,k]),pct))
    columns=["Condition","Model","Candidate PV","Target PV","Assumed measurement noise pp",
             "Current SD pp","Approx variance reduction pct"]
    pd.DataFrame(gains,columns=columns).to_csv(root/"design_sampling_gain.csv",index=False)


    observed_intervals=EDGES[:-1]<PRIMARY
    slow=np.where(observed_intervals,.24,.025)
    fast=np.where(observed_intervals,.24,.17)
    assert np.max(np.r_[slow,fast])<=1 and max(WIDTH@slow,WIDTH@fast)<=1
    x=np.unique(np.r_[np.linspace(0,EDGES[-1],321),EDGES,PRIMARY,TARGET])
    low,high=100*(np.stack([slow,fast])@recovery_design(x).T)
    assert np.max(abs(low[x<=PRIMARY]-high[x<=PRIMARY]))==0
    pd.DataFrame({"PV":x,"Recovery A pp":low,"Recovery B pp":high,
                  "RF gap pp":high-low}).to_csv(root/"design_counterexample.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    run(p.parse_args().results)
