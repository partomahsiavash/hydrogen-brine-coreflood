import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import platform

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.svm import SVR
from demo_config import ORIGINS, PRIMARY, TEST, CHECK_ORIGIN
from synthetic_inputs import make_observations, balance_audit, simulate_innovations, synthetic_grid
from inverse import (EDGES, WIDTH, SLOPE_SD, FOCAL, Fit, Series, diagnostics, prior_draws, pressure_design,
                     recovery_design, whiten)

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Scenario:
    name: str
    rf: bool = True
    pin: bool = False
    strict: bool = True
    linked: bool = False
    tail_width: float = 1.
    lengths: tuple = (.025,.035)
    robust: bool = False
    stride: int = 4


RF_BOUND = Scenario("RF bounded")
RF_FREE = Scenario("RF flexible", strict=False)
J_BOUND = Scenario("Joint bounded", pin=True, linked=True)
J_FREE = Scenario("Joint flexible", pin=True, linked=True, strict=False)
P_ONLY = Scenario("Pressure only", rf=False, pin=True)
STANDARD = (RF_BOUND,RF_FREE,J_BOUND,J_FREE,P_ONLY)
SENSITIVITIES = (
    Scenario("RF bounded narrow tail", tail_width=.5),
    Scenario("RF bounded wide tail", tail_width=2),
    Scenario("RF flexible narrow tail", strict=False, tail_width=.5),
    Scenario("RF flexible wide tail", strict=False, tail_width=2),
    Scenario("Joint bounded Student t", pin=True, linked=True, robust=True),
    Scenario("Joint flexible Student t", pin=True, linked=True, strict=False, robust=True),
    Scenario("Pressure only Student t", rf=False, pin=True, robust=True),
    Scenario("Joint flexible alternate OU", pin=True, linked=True, strict=False, lengths=(.05,.07),stride=2),
    Scenario("Pressure only alternate OU", rf=False, pin=True, lengths=(.05,.07),stride=2),
)


def get_training(observed, channel, origin, length, stride):
    frame = observed[(observed.Channel == channel) & (observed.PV <= origin+1e-10)]
    y = frame.Observed.to_numpy()
    scale, offset = (100.,0.) if channel == "RF" else (float(np.ptp(y)), float(y.min()))
    if len(y)<3 or scale==0:
        raise ValueError("Insufficient training observations or pressure range.")
    idx = np.unique(np.r_[np.arange(0,len(y),stride),len(y)-1])
    return Series(frame.PV.to_numpy()[idx], (y[idx]-offset)/scale, length, scale, offset)


def forecast_rows(condition, origin, scenario, observed, model, chains, keys, seed):
    tables = []
    for channel, data in (("RF",model.rf),("Pinlet",model.pin)):
        if data is None:
            continue
        q = observed[(observed.Channel == channel) & (observed.PV > origin+1e-10)]
        x, y = q.PV.to_numpy(), q.Observed.to_numpy()
        mean, predictive = model.predictive(chains,keys,x,channel,seed+int(channel=="Pinlet"))
        a,b,c = np.quantile(predictive,[.05,.50,.95],axis=0)
        lo,mid,hi = np.quantile(mean,[.05,.50,.95],axis=0)
        sorted_draws = np.sort(predictive,axis=0)
        m = len(sorted_draws)
        weights = (2*np.arange(1,m+1)-m-1)/(m*m)
        crps = np.abs(predictive-y).mean(axis=0) - weights @ sorted_draws
        interval = c-a + 20*(np.maximum(a-y,0)+np.maximum(y-c,0))
        tables.append(pd.DataFrame({"Condition":condition,"Origin":origin,"Model":scenario,
                        "Channel":channel,"PV":x,"Observed":y,"Forecast":b,"P05":a,"P95":c,
                        "Latent 05":lo,"Latent 50":mid,"Latent 95":hi,"CRPS":crps,
                        "Interval score 90":interval,"Source row":q["Source row"].to_numpy(),
                        "Training observations":len(data.x)}))
    return pd.concat(tables,ignore_index=True)


def local_pressure(observed, model, chains, keys):
    if not model.linked:
        return []
    md = np.median(chains,axis=(0,1))
    rf, pin = model.rf, model.pin
    gamma = md[keys.index("gamma")]
    whiten_rf = whiten(rf.x,recovery_design(rf.x),rf.length)/md[keys.index("rf_sigma")]
    whiten_rec = whiten(pin.x,4*recovery_design(pin.x),pin.length)/md[keys.index("pin_sigma")]
    whiten_pin = whiten(pin.x,pressure_design(pin.x),pin.length)/md[keys.index("pin_sigma")]
    nuisance = np.column_stack((whiten_pin,whiten_rec@md[:len(WIDTH)]))
    u,s,_ = np.linalg.svd(nuisance,full_matrices=False)
    active = u[:,s>max(nuisance.shape)*np.finfo(float).eps*s[0]]
    coupling = whiten_rec*gamma
    projected = coupling - active @ (active.T@coupling)
    _,s0,v = np.linalg.svd(whiten_rf*SLOPE_SD,full_matrices=False)
    moves = projected*SLOPE_SD @ v.T
    gain = np.sqrt(s0*s0+(moves*moves).sum(0))
    out = []
    for i,(r,j) in enumerate(zip(s0,gain),1):
        out.append((i,r,j,100*(j/r-1) if r>1e-8 else np.nan))
    return out


def fit_task(task):
    condition, observed, origin, cfg, seed, burn, draws = task
    rf = get_training(observed,"RF",origin,cfg.lengths[0],1) if cfg.rf else None
    pin = get_training(observed,"Pinlet",origin,cfg.lengths[1],cfg.stride) if cfg.pin else None
    model = Fit(rf,pin,cfg.strict,cfg.linked,cfg.tail_width,cfg.robust)
    chain, keys = model.sample(burn,draws,seed)
    if rf is not None:
        r = chain[..., :len(WIDTH)]
        assert np.min(r)>=0 and np.max(r@WIDTH)<=1+1e-10
        assert not cfg.strict or np.max(r)<=1+1e-10
    prediction = forecast_rows(condition,origin,cfg.name,observed,model,chain,keys,seed+2718)
    dg = [(condition,origin,cfg.name,*row) for row in diagnostics(chain,keys)]
    sens = [(condition,origin,cfg.name,*row) for row in local_pressure(observed,model,chain,keys)]
    priors = []
    if cfg.rf:
        p = prior_draws(cfg.tail_width,cfg.strict,seed+8901)
        design = recovery_design(FOCAL)
        p_q = np.quantile(p@design.T,[.05,.50,.95],axis=0)
        q_q = np.quantile(chain[...,:len(WIDTH)].reshape(-1,len(WIDTH))@design.T,[.05,.50,.95],axis=0)
        for x,prior,post in zip(FOCAL,p_q.T,q_q.T):
            priors.append((condition,origin,cfg.name,x,*(prior*100),*(post*100),
                           (post[2]-post[0])/(prior[2]-prior[0])))
    print(condition,origin,cfg.name,flush=True)
    return prediction,dg,sens,priors,chain,keys


def weibull(x,p):
    cap,onset,scale,shape = p
    return cap*(-np.expm1(-(np.maximum(0,x-onset)/scale)**shape))


def point_forecasts(condition, observed, origin):
    tables = []
    for channel in ("RF","Pinlet"):
        train = get_training(observed,channel,origin,0.,1 if channel=="RF" else 4)
        h = observed[(observed.Channel == channel) & (observed.PV > origin+1e-10)]
        x = h.PV.to_numpy()
        scaler = train.x[-1]
        svr = SVR(C=3.,epsilon=.015,gamma=3.).fit(train.x[:,None]/scaler,train.y)
        models = {"Persistence":np.full(len(x), train.y[-1]),
                  "RBF SVR":svr.predict(x[:,None]/scaler)}
        if channel == "RF":
            fits = [least_squares(lambda p: weibull(train.x,p)-train.y,start,
                    bounds=([1e-6,0.,.003,.55],[1.,.39,3.5,8.]),max_nfev=400)
                    for start in ([.25,.24,.2,2.],[.35,.14,.4,3.],[.22,.30,.08,1.])]
            models["Weibull least squares"] = weibull(x,min(fits,key=lambda f:f.cost).x)
        for name,y in models.items():
            tables.append(pd.DataFrame({"Condition":condition,"Origin":origin,"Model":name,
                "Channel":channel,"PV":x,"Observed":h.Observed.to_numpy(),
                "Forecast":y*train.scale+train.offset,"Source row":h["Source row"].to_numpy(),
                "Training observations":len(train.x)}))
    return pd.concat(tables,ignore_index=True)


def summarize(pred):
    work = pred.assign(Window=np.where(pred.PV<=TEST+1e-10,"Intermediate","Late"))
    metrics = []
    for ids,g in work.groupby(["Condition","Origin","Model","Channel","Window"],sort=True):
        error = g.Forecast-g.Observed
        hits = int(((g.P05<=g.Observed)&(g.Observed<=g.P95)).sum()) if g.P05.notna().all() else np.nan
        metrics.append((*ids,len(g),np.sqrt(np.mean(error**2)),np.abs(error).mean(),error.mean(),
                        g.CRPS.mean(),hits,hits/len(g),(g.P95-g.P05).mean(),g["Interval score 90"].mean()))
    return pd.DataFrame(metrics,columns=["Condition","Origin","Model","Channel","Window","n",
                     "RMSE","MAE","Bias predicted minus observed","CRPS","Hits 90","Coverage 90",
                     "Width 90","Interval score 90"])


def make_synthetic(noise_seed=8712):
    rates = np.array([.32, .48, .31, .20, .14, .10, .08, .06, .04, .02])
    configs = []
    for gamma in (0., .9):
        rng = np.random.default_rng(noise_seed)
        tag = f"Synthetic gamma {gamma:.1f}"
        rows = []
        for channel, x in synthetic_grid().items():
            r = recovery_design(x) @ rates
            if channel == "RF":
                y = 100 * (r + simulate_innovations(x, .010, .04, rng))
            else:
                trend = pressure_design(x) @ np.array([.15, .19, .32, .19, .025])
                y = 60 * (trend + 4*gamma*r + simulate_innovations(x, .04, .06, rng))
            rows.extend((tag, channel, a, b, j)
                        for j, (a, b) in enumerate(zip(x, y), 1))
        data = pd.DataFrame(rows, columns=["Condition", "Channel", "PV", "Observed", "Source row"])
        configs.append((tag, data, gamma, rates))
    return configs


def main(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    observations = make_observations(args.observation_seed)
    conditions = [(c,frame) for c,frame in observations.groupby("Condition",sort=False)]
    system = [(c,frame,origin,cfg) for c,frame in conditions for origin in ORIGINS for cfg in STANDARD]
    system += [(c,frame,PRIMARY,cfg) for c,frame in conditions for cfg in SENSITIVITIES]
    synthetic = make_synthetic()
    system += [(c,frame,CHECK_ORIGIN,cfg) for c,frame,_,_ in synthetic for cfg in (RF_BOUND,J_BOUND,P_ONLY)]
    tasks = [(*entry,args.seed+i*29,args.burn,args.draws) for i,entry in enumerate(system)]
    estimates, diag, sensitivity, contraction, posterior, slots = [],[],[],[],{},[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (spec, result) in enumerate(zip(system,pool.map(fit_task,tasks))):
            condition,_,origin,cfg = spec
            prediction,dg,sens,prior,chain,keys = result
            estimates.append(prediction)
            diag.extend(dg)
            sensitivity.extend(sens)
            contraction.extend(prior)
            if origin == PRIMARY or condition.startswith("Synthetic"):
                key = f"fit_{i:03d}"
                posterior[key] = chain.astype("float64",copy=False)
                slots.append(dict(array=key,condition=condition,origin=origin,scenario=asdict(cfg),columns=keys))
    estimates.extend(point_forecasts(c,frame,origin) for c,frame in conditions for origin in ORIGINS)
    pred = pd.concat(estimates,ignore_index=True)
    metrics = summarize(pred)
    diagnostics_df = pd.DataFrame(diag,columns=["Condition","Origin","Model","Quantity","Q05","Median",
                                                 "Q95","Rhat","ESS bulk","ESS tail","Pass"])
    sensitivities = pd.DataFrame(sensitivity,columns=["Condition","Origin","Model","Direction","RF norm",
                                                     "Joint projected norm","Gain pct"])
    contractions = pd.DataFrame(contraction,columns=["Condition","Origin","Model","PV","Prior 05",
                "Prior 50","Prior 95","Posterior 05","Posterior 50","Posterior 95","Post prior width ratio"])
    truth_rows=[]
    for condition,frame,gamma,rates in synthetic:
        s=get_training(frame,"Pinlet",CHECK_ORIGIN,.035,4)
        for x,r in zip(FOCAL,recovery_design(FOCAL)@rates):
            truth_rows.append((condition,x,r*100,gamma,60*gamma/s.scale))
    truth = pd.DataFrame(truth_rows,columns=["Condition","PV","True recovery pp","Raw simulated gamma",
                                           "True normalized gamma"])
    manifest = {"seed":args.seed,"observation_seed":args.observation_seed,"burn":args.burn,"draws":args.draws,"chains":4,"origins":ORIGINS,
                "reference_origin":PRIMARY,"late_boundary":TEST,"numpy":np.__version__,
                "pandas":pd.__version__,"python":platform.python_version(),"edges":EDGES.tolist(),
                "slope_prior_sd":SLOPE_SD.tolist(),"posterior_arrays":slots}
    true_inputs = pd.DataFrame([(c,origin,chan,get_training(frame,chan,origin,l,s).scale,
                                 get_training(frame,chan,origin,l,s).offset)
                                for c,frame in conditions for origin in ORIGINS
                                for chan,l,s in (("RF",.025,1),("Pinlet",.035,4))],
                               columns=["Condition","Origin","Channel","Training scale","Training offset"])
    sheets={"Metrics":metrics,"Diagnostics":diagnostics_df,"Prior and posterior":contractions,
            "Pressure sensitivity":sensitivities,"Synthetic truth":truth,"Forecasts":pred,
            "Source observations":observations,"Volume audit":balance_audit(observations),
            "Training scales":true_inputs,
            "Rate grid":pd.DataFrame({"Start":EDGES[:-1],"End":EDGES[1:],"Prior SD":SLOPE_SD})}
    for name,table in sheets.items():
        table.to_csv(out/(name.lower().replace(" ","_")+".csv"),index=False)
    np.savez_compressed(out/"posterior_chains.npz",**posterior)
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(metrics.query("Origin == @PRIMARY and Window == 'Late'").round(4).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-seed",type=int,default=6017)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--seed",type=int,default=216713)
    parser.add_argument("--burn",type=int,default=1500)
    parser.add_argument("--draws",type=int,default=3500)
    parser.add_argument("--workers",type=int,default=min(5,os.cpu_count() or 1))
    args=parser.parse_args()
    if min(args.burn,args.draws,args.workers)<1:
        parser.error("burn, draws and workers must be positive")
    main(args)
