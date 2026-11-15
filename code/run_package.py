import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from export_results import prepare_tables, write_workbook
from result_tables import FILENAME

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True,cwd=ROOT)


def main(args):
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="h2_coreflood_",dir=output.parent) as work:
        staged=Path(work)
        charts=staged/"figures"
        run("run_advanced.py","--observation-seed",args.observation_seed,"--output",staged,"--workers",args.workers,
            "--seed",args.seed,"--burn",args.burn,"--draws",args.draws)
        run("synthetic_repeats.py","--output",staged,"--workers",args.workers,
            "--replicates",args.synthetic_replicates,"--burn",args.synthetic_burn,"--draws",args.synthetic_draws)
        run("design_audit.py","--results",staged)
        run("verify_results.py","--results",staged)
        run("plot_advanced.py","--results",staged,"--dest",charts)
        write_workbook(prepare_tables(staged),staged/FILENAME)
        output.mkdir(parents=True,exist_ok=True)
        default_figures=ROOT/"figures" if output==ROOT/"results" else output/"figures"
        chart_dir=args.figures.resolve() if args.figures else default_figures
        chart_dir.mkdir(parents=True,exist_ok=True)
        for name in (FILENAME,"posterior_chains.npz","run_manifest.json"):
            shutil.move(staged/name,output/name)
        for plot in charts.iterdir():
            shutil.move(plot,chart_dir/plot.name)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run core-flood inference and export a single numerical workbook.")
    parser.add_argument("--observation-seed",type=int,default=6017)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=216713)
    parser.add_argument("--burn",type=int,default=1500)
    parser.add_argument("--draws",type=int,default=3500)
    parser.add_argument("--synthetic-replicates",type=int,default=12)
    parser.add_argument("--synthetic-burn",type=int,default=750)
    parser.add_argument("--synthetic-draws",type=int,default=1750)
    main(parser.parse_args())
