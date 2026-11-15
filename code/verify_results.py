import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from result_tables import read_result, TABLES

from inverse import WIDTH
from demo_config import TEST
from synthetic_inputs import make_observations


def main(results):
    archive=json.loads((results/"run_manifest.json").read_text())
    expected=make_observations(archive["observation_seed"])
    actual=read_result(results,"source_observations")
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False)
    metrics=read_result(results,"metrics")
    forecast=read_result(results,"forecasts")
    assert (forecast.PV>forecast.Origin).all()
    forecast["Window"]=np.where(forecast.PV<=TEST+1e-10,"Intermediate","Late")
    keys=["Condition","Origin","Model","Channel","Window"]
    score=metrics.set_index(keys)
    for k,g in forecast.groupby(keys,sort=False):
        r=score.loc[k]
        assert len(g)==r.n
        np.testing.assert_allclose(np.sqrt(np.mean((g.Forecast-g.Observed)**2)),r.RMSE,atol=1e-11)
        if g.P05.notna().all():
            assert (g.P05<=g.Forecast).all() and (g.Forecast<=g.P95).all()
            assert int(((g.Observed>=g.P05)&(g.Observed<=g.P95)).sum())==r["Hits 90"]
            assert (g.CRPS>=-1e-10).all()
    for _,g in forecast[(forecast.Window=="Late")&(~forecast.Condition.str.startswith("Synthetic"))].groupby(
            ["Condition","Channel"]):
        groups=[tuple(x.PV) for _,x in g.groupby(["Origin","Model"])]
        assert all(candidate==groups[0] for candidate in groups)
    checked=0
    with np.load(results/"posterior_chains.npz") as npz:
        for fit in archive["posterior_arrays"]:
            chain=npz[fit["array"]]
            assert chain.shape==(archive["chains"],archive["draws"],len(fit["columns"]))
            assert np.isfinite(chain).all()
            if fit["scenario"]["rf"]:
                rates=chain[...,:len(WIDTH)]
                assert rates.min()>=0 and (rates @ WIDTH).max()<=1+1e-12
                assert not fit["scenario"]["strict"] or rates.max()<=1+1e-12
            checked+=1
    d=read_result(results,"diagnostics")
    rows={"synthetic_observations_verified":len(expected),"metric_rows_verified":len(metrics),
          "forecast_rows_verified":len(forecast),"archived_chains_verified":checked,"worksheets":len(TABLES)+1,
          "diagnostics_combinations":len(d),"diagnostics_review_count":int((~d.Pass).sum()),
          "worst_rhat":float(d.Rhat.max()),"min_bulk_ess":float(d["ESS bulk"].min()),
          "min_tail_ess":float(d["ESS tail"].min())}
    (results/"verification.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    args=p.parse_args()
    main(args.results)
