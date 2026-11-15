import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from inverse import recovery_design, WIDTH
from result_tables import read_result

from demo_config import CASES as FLUIDS, PRIMARY as ORIGIN, TEST, TARGET
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8B499A", "#648491", "#4B5563")
RF_MODELS = ("RF bounded", "RF flexible", "Joint bounded", "Joint flexible",
             "Weibull least squares", "Persistence")
PIN_MODELS = ("Pressure only", "Joint bounded", "Joint flexible",
              "Pressure only Student t", "Joint flexible Student t", "Persistence")
RF_LABELS = ("RF\nbounded", "RF\nflexible", "Joint\nbounded", "Joint\nflexible", "Weibull", "Persistence")
PIN_LABELS = ("Pressure\nonly", "Joint\nbounded", "Joint\nflexible", "Pressure\nt", "Joint\nt", "Persistence")


def canvas(width=8.8, height=5.9):
    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    fig.subplots_adjust(left=.09, right=.975, top=.88, bottom=.14, wspace=.26, hspace=.57)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-.08, 1.08, f"({label})", transform=ax.transAxes, fontweight="bold")
        ax.grid(axis="y", alpha=.24)
        ax.set_axisbelow(True)
    return fig, axes


def save(fig, dest, name):
    for extension in ("svg", "pdf", "png"):
        fig.savefig(dest / f"{name}.{extension}", dpi=420)
    plt.close(fig)


def error_boxplots(results, dest):
    pred = read_result(results, "forecasts")
    pred = pred[(pred.Origin==ORIGIN) & (pred.PV>TEST)]
    fig, axes = canvas()
    for row, (channel, names, labels) in enumerate((
            ("RF",RF_MODELS,RF_LABELS),("Pinlet",PIN_MODELS,PIN_LABELS))):
        bounds=[]
        for col, condition in enumerate(FLUIDS):
            ax=axes[row,col]
            table=pred[(pred.Condition==condition) & (pred.Channel==channel)]
            sample=[(g.Forecast-g.Observed).to_numpy() for name in names
                    for g in [table[table.Model==name]]]
            if any(len(values)==0 for values in sample):
                raise ValueError(f"Missing error samples for {condition} {channel}")
            boxes=ax.boxplot(sample,widths=.54,patch_artist=True,showfliers=False,
                             medianprops={"color":"#151F2B","linewidth":1.45},
                             whiskerprops={"color":"#657384"},capprops={"color":"#657384"})
            rng=np.random.default_rng(749+row*7+col)
            for i,(points,patch,color) in enumerate(zip(sample,boxes["boxes"],COLORS),1):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(.27)
                ax.scatter(i+rng.uniform(-.17,.17,len(points)),points,
                           s=18 if row==0 else 4.2,alpha=.84 if row==0 else .16,
                           linewidths=0,color=color,zorder=4)
            ax.axhline(0,color="#657384",ls="--",lw=.85)
            ax.set_xticks(range(1,len(names)+1),labels,fontsize=7.5)
            ax.tick_params(axis="x",length=0,pad=5)
            ax.set_title(f"{condition} · {'recovery' if row==0 else 'inlet pressure'}")
            ax.set_ylabel("Signed error (pp)" if row==0 else "Signed error (arb. units)")
            bounds.extend(np.concatenate(sample))
        low=min(min(bounds),0)
        high=max(max(bounds),0)
        margin=.12*max(high-low,1.)
        for ax in axes[row]:
            ax.set_ylim(low-margin,high+margin)
    fig.suptitle("Held-out forecast error distributions",fontsize=14,fontweight="bold",y=.985)
    fig.text(.5,.045,f"Forecast − simulated observation | Training ≤ {ORIGIN:.2f} PV · Test > {TEST:.2f} PV",ha="center")
    fig.text(.5,.019,"Dots show held-out simulated observations. Pressure errors are serially correlated.",ha="center",fontsize=7.6)
    save(fig,dest,"Fig_05_Heldout_error_boxplots")


def posterior_violins(results, dest):
    manifest=json.loads((results/"run_manifest.json").read_text(encoding="utf-8"))
    names=("RF bounded narrow tail","RF bounded","RF bounded wide tail")
    labels=("0.5× prior SD","1× prior SD","2× prior SD")
    locations=(ORIGIN,TARGET)
    fig,axes=canvas()
    with np.load(results/"posterior_chains.npz") as posterior:
        for row,x in enumerate(locations):
            limits=[]
            for col,condition in enumerate(FLUIDS):
                ax=axes[row,col]
                design=recovery_design(np.array([x]))[0]
                sample=[]
                for name in names:
                    spec=next(f for f in manifest["posterior_arrays"] if f["condition"]==condition
                              and f["scenario"]["name"]==name and f["origin"]==ORIGIN)
                    draws=posterior[spec["array"]][...,:len(WIDTH)].reshape(-1,len(WIDTH))
                    sample.append(100*(draws@design))
                violins=ax.violinplot(sample,showmeans=False,showextrema=False,showmedians=False,widths=.70,
                                      points=150)
                for i,(body,draws,color) in enumerate(zip(violins["bodies"],sample,COLORS),1):
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(.22)
                    lo,md,hi=np.quantile(draws,[.05,.5,.95])
                    ax.vlines(i,lo,hi,lw=2.2,color=color)
                    ax.scatter(i,md,facecolor="white",edgecolor=color,s=29,lw=1.3,zorder=5)
                    ax.annotate(f"{md:.1f}%",(i,md),xytext=(6,1),textcoords="offset points",
                                fontsize=7.75,va="bottom",color="#192A3C")
                ax.set_xticks([1,2,3],labels,fontsize=8)
                ax.set_ylabel("Latent liquid recovery (%)")
                ax.set_title(f"{condition} · {x:.2f} PV")
                ax.set_xlim(.5,3.65)
                limits.extend(np.concatenate(sample))
            low,high=min(limits),max(limits)
            padding=.06*max(high-low,1.)
            for ax in axes[row]:
                ax.set_ylim(max(0,low-padding),min(100,high+padding))
    fig.suptitle("Posterior recovery across late-rate priors",fontsize=14,fontweight="bold",y=.985)
    draws_per_fit = manifest["chains"] * manifest["draws"]
    fig.text(.5,.042,f"Slope-bounded recovery-only model · {draws_per_fit:,} retained MCMC draws per fit",
             ha="center",fontsize=8.35)
    fig.text(.5,.017,"White markers are posterior medians and vertical bars are 90% credible intervals. Rows use different vertical scales.",
             ha="center",fontsize=7.6)
    save(fig,dest,"Fig_06_Posterior_recovery_violins")


def predictive_scorecards(results, dest):
    metrics=read_result(results,"metrics")
    rows=metrics[(metrics.Origin==ORIGIN)&(metrics.Window=="Late")]
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.35))
    fig.subplots_adjust(left=.145,right=.98,top=.85,bottom=.115,wspace=.63,hspace=.72)
    cmap=LinearSegmentedColormap.from_list("scores",["#ACD9DD","#EDF5F7","#FAE7D4"])
    fields=("CRPS","Interval score 90","Width 90","Hits 90")
    groups=(("RF",RF_MODELS[:4],("RF bounded","RF flexible","Joint bounded","Joint flexible")),
            ("Pinlet",PIN_MODELS[:5],("Pressure only","Joint bounded","Joint flexible","Pressure only t","Joint flexible t")))
    for row,(channel,names,labels) in enumerate(groups):
        common=rows[(rows.Channel==channel)&rows.Model.isin(names)]
        scale={}
        for field in fields[:2]:
            scale[field]=(common[field].min(),common[field].max())
        for col,condition in enumerate(FLUIDS):
            ax=axes[row,col]
            t=common[common.Condition==condition].set_index("Model").loc[list(names)]
            for i,(_,line) in enumerate(t.iterrows()):
                for j,field in enumerate(fields):
                    value=line[field]
                    if field in scale:
                        lo,hi=scale[field]
                        fill=cmap((value-lo)/max(hi-lo,1e-12))
                    else:
                        fill="#F1F4F7"
                    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=fill,ec="white",lw=2))
                    label=(f"{int(value)} / {int(line['n'])}" if field=="Hits 90" else f"{value:.2f}")
                    ax.text(j,i,label,ha="center",va="center",fontsize=9,color="#172839")
            ax.text(-.04,1.24,f"({'abcd'[row*2+col]})",transform=ax.transAxes,fontweight="bold")
            unit="pp" if channel=="RF" else "arb. units"
            ax.set_title(f"{condition} · {unit}",pad=29)
            ax.set_xlim(-.5,3.5)
            ax.set_ylim(len(names)-.5,-.5)
            ax.set_xticks(range(4),["CRPS ↓","Interval\nscore ↓","90% width","90% hits"],
                          fontsize=8)
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(names)),labels,fontsize=7.9)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
    fig.suptitle("Probabilistic forecast comparison",fontweight="bold",fontsize=14,y=.985)
    fig.text(.5,.051,f"Training ≤ {ORIGIN:.2f} PV · Test > {TEST:.2f} PV · Blue marks smaller scores within each output and metric.",
             ha="center",fontsize=8.25)
    fig.text(.5,.022,"Widths and hit counts are descriptive; observed coverage is not a calibration test.",
             ha="center",fontsize=7.65)
    save(fig,dest,"Fig_07_Probabilistic_forecast_scorecards")
