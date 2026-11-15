import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from result_tables import read_result

from demo_config import CASES as FLUIDS, PRIMARY, ORIGINS, TEST
COLOR={"RF bounded":"#0072B2","RF flexible":"#D55E00","Joint bounded":"#009E73",
       "Joint flexible":"#8B499A","Pressure only":"#0072B2","Persistence":"#555555",
       "Weibull least squares":"#A57400","RBF SVR":"#949494"}
STYLE={"RF bounded":"-","RF flexible":"--","Joint bounded":"-.","Joint flexible":"--",
       "Pressure only":"-","Persistence":":","Weibull least squares":"-.","RBF SVR":"-"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.labelsize":8.5,
    "axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"svg.fonttype":"none","grid.alpha":.17,"axes.axisbelow":True})


def canvas():
    f,a=plt.subplots(2,2,figsize=(7.6,5.4))
    f.subplots_adjust(left=.11,right=.97,bottom=.16,top=.94,wspace=.37,hspace=.55)
    for label,ax in zip("abcd",a.ravel()):
        ax.text(-.10,1.06,f"({label})",transform=ax.transAxes,fontweight="bold")
        ax.grid(axis="y")
    return f,a


def save(fig, out, name):
    for kind in ["svg","pdf","png"]:
        fig.savefig(out/f"{name}.{kind}",dpi=420)
    plt.close(fig)


def key(ax, x, y, name, **extra):
    return ax.plot(x,y,label=name,color=COLOR[name],ls=STYLE[name],**extra)[0]


def forecast(results,dest):
    pred=read_result(results,"forecasts")
    obs=read_result(results,"source_observations")
    fig,a=canvas()
    for i,c in enumerate(FLUIDS):
        for j,channel in enumerate(("RF","Pinlet")):
            ax=a[j,i]
            q=pred[(pred.Condition==c)&(pred.Channel==channel)&(pred.Origin==PRIMARY)]
            raw=obs[(obs.Condition==c)&(obs.Channel==channel)]
            models=("RF bounded","RF flexible","Persistence") if j==0 else (
                "Pressure only","Joint flexible","Persistence")
            focus=q[q.Model==models[0]]
            ax.axvspan(0,PRIMARY,color="#EEF1F5")
            ax.axvline(TEST,color="#999999",ls="--",lw=.8)
            ax.fill_between(focus.PV,focus.P05,focus.P95,color=COLOR[models[0]],alpha=.16,linewidth=0)
            for name in models:
                p=q[q.Model==name]
                key(ax,p.PV,p.Forecast,name,lw=1.4)
            if j==0:
                ax.scatter(raw.PV,raw.Observed,s=11,c="#181818",zorder=5)
                ax.set(title=c,ylabel="Liquid recovery (%)",ylim=(-4,52))
            else:
                ax.plot(raw.PV,raw.Observed,lw=.85,c="#181818",zorder=5)
                ax.set_ylabel("Inlet pressure (arb. units)")
            ax.set_xlabel("Injected pore volumes (PV)")
    names=["RF bounded","RF flexible","Pressure only","Joint flexible","Persistence"]
    handles=[Line2D([],[],label=n,color=COLOR[n],ls=STYLE[n]) for n in names]
    handles.append(Line2D([],[],color="#181818",marker="o",ms=3,label="Simulated observation"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.022),ncol=3,frameon=False)
    save(fig,dest,"Fig_01_Conditional_forecasts")


def identifiability(results,dest):
    p=read_result(results,"prior_and_posterior")
    blind=read_result(results,"design_counterexample")
    gain=read_result(results,"design_sampling_gain")
    fig,a=canvas()
    hues=["#0072B2","#009E73","#D55E00"]
    settings=["RF bounded narrow tail","RF bounded","RF bounded wide tail"]
    for i,c in enumerate(FLUIDS):
        ax=a[0,i]
        q=p[(p.Condition==c)&(p.Origin==PRIMARY)&p.Model.isin(settings)]
        for name,color,label in zip(settings,hues,["0.5× tail prior","1× tail prior","2× tail prior"]):
            f=q[q.Model==name]
            ax.fill_between(f.PV,f["Posterior 05"],f["Posterior 95"],color=color,alpha=.09)
            ax.plot(f.PV,f["Posterior 50"],color=color,marker="o",ms=3,label=label)
        f=q[q.Model=="RF bounded"]
        ax.plot(f.PV,f["Prior 50"],ls="--",lw=1.1,color="#626262",label="Prior median (1×)")
        ax.axvline(PRIMARY,color="#999",lw=.75)
        ax.set(title=c,xlabel="Prediction PV",ylabel="Latent recovery (%)",ylim=(0,82))
    ax=a[1,0]
    ax.plot(blind.PV,blind["Recovery A pp"],color=hues[0],lw=1.4,label="Admissible A")
    ax.plot(blind.PV,blind["Recovery B pp"],color=hues[2],lw=1.4,label="Admissible B")
    ax.axvspan(0,PRIMARY,color="#EEF1F5")
    ax.set(xlabel="Injected pore volumes (PV)",ylabel="Liquid recovery (%)",title="Identical training trajectories")
    ax.legend(frameon=False,loc="upper left",fontsize=7.4)
    ax=a[1,1]
    for c,color in zip(FLUIDS,[hues[0],hues[2]]):
        for sd,style in zip((1,2),["-","--"]):
            f=gain[(gain.Condition==c)&(gain.Model=="RF bounded")&(gain["Target PV"]==2)&
                   (gain["Assumed measurement noise pp"]==sd)]
            ax.plot(f["Candidate PV"],f["Approx variance reduction pct"],color=color,ls=style,
                    label=f"{c}, {sd} pp")
    ax.set(title="Additional recovery measurement",xlabel="Candidate measurement PV",
           ylabel="RF at 2 PV variance reduction (%)")
    ax.legend(fontsize=6.4,loc="best",frameon=False)
    fig.legend(*a[0,0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.030),
               ncol=4,frameon=False,fontsize=7.5)
    save(fig,dest,"Fig_02_Identifiability_and_design")


def robustness(results,dest):
    m=read_result(results,"metrics")
    m=m[m.Window=="Late"]
    fig,a=canvas()
    for i,c in enumerate(FLUIDS):
        for j,channel in enumerate(("RF","Pinlet")):
            ax=a[j,i]
            q=m[(m.Condition==c)&(m.Channel==channel)]
            names=("RF bounded","RF flexible","Joint bounded","Joint flexible","Persistence",
                   "Weibull least squares") if j==0 else ("Pressure only","Joint bounded","Joint flexible",
                                                        "Persistence")
            for name in names:
                w=q[q.Model==name]
                key(ax,w.Origin,w.RMSE,name,lw=1.25,marker="o",ms=3)
            ax.set(xticks=ORIGINS,xlabel="Training cutoff (PV)",
                   ylabel="RMSE (pp)" if j==0 else "RMSE (arb. units)")
            if j==0:
                ax.set_title(c)
    hs,ls=a[0,0].get_legend_handles_labels()
    h,l=a[1,0].get_legend_handles_labels()
    fig.legend(hs+[h[0]],ls+[l[0]],loc="lower center",bbox_to_anchor=(.5,.015),
               frameon=False,ncol=4,fontsize=7.4)
    save(fig,dest,"Fig_03_Blocked_robustness")


def synthetic(results,dest):
    d=read_result(results,"diagnostics")
    t=read_result(results,"synthetic_truth")
    q=d[d.Quantity=="gamma"].copy()
    q=q[(q.Origin==PRIMARY)&q.Model.isin(["Joint bounded","Joint flexible"])|
        (q.Condition.str.startswith("Synthetic") & (q.Model=="Joint bounded"))]
    order=[(FLUIDS[0],"Joint bounded"),(FLUIDS[0],"Joint flexible"),
           (FLUIDS[1],"Joint bounded"),(FLUIDS[1],"Joint flexible"),
           ("Synthetic gamma 0.0","Joint bounded"),("Synthetic gamma 0.9","Joint bounded")]
    q=q.set_index(["Condition","Model"]).loc[order].reset_index()
    fig,ax=plt.subplots(figsize=(7.6,3.0))
    fig.subplots_adjust(left=.285,right=.96,top=.95,bottom=.19)
    at=np.arange(len(q))
    ax.axvline(0,color="#888",ls="--",lw=.8)
    ax.errorbar(q.Median,at,xerr=(q.Median-q.Q05,q.Q95-q.Median),fmt="o",ms=4,
                color="#0072B2",capsize=2,label="90% posterior")
    truth=t.groupby("Condition")["True normalized gamma"].first()
    ax.scatter([truth[c] for c in q.Condition[4:]],at[4:],marker="D",s=28,c="#D55E00",
               label="Simulated truth",zorder=5)
    ax.set_yticks(at,["Example A, bounded","Example A, flexible","Example B, bounded","Example B, flexible",
                     "Synthetic, zero link","Synthetic, nonzero link"])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized coupling coefficient γ")
    ax.legend(loc="upper right",frameon=False,fontsize=7.6)
    save(fig,dest,"Fig_04_Pressure_coupling")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    p.add_argument("--dest",type=Path,default=Path("figures"))
    args=p.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    from plot_comparisons import error_boxplots, posterior_violins, predictive_scorecards
    for plot in (forecast,identifiability,robustness,synthetic,error_boxplots,posterior_violins,predictive_scorecards):
        plot(args.results,args.dest)
