"""Summarize M9 load-basis sensitivity results and plot acceptance."""
from __future__ import annotations
import argparse, csv, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('results/m9_load_basis_audit'))
    args=parser.parse_args()
    rows=list(csv.DictReader((args.input_dir/'m9_load_basis_runs.csv').open(newline='',encoding='utf-8')))
    policies=sorted({r['policy'] for r in rows})
    bases=('paper_slice_units','compute','memory','max_normalized')
    summary=[]
    for basis in bases:
        for policy in policies:
            group=[r for r in rows if r['load_basis']==basis and r['policy']==policy]
            if not group: continue
            rec={'load_basis':basis,'policy':policy,'runs':len(group)}
            for metric in ('acceptance_rate','offered_paper_units_fraction','offered_compute_fraction','offered_memory_fraction','peak_arrival_compute_utilization','peak_arrival_memory_utilization','mean_decision_time_ns'):
                vals=[float(r[metric]) for r in group]
                rec[f'{metric}_mean']=statistics.mean(vals)
                rec[f'{metric}_sd']=statistics.stdev(vals) if len(vals)>1 else 0.0
            summary.append(rec)
    with (args.input_dir/'m9_load_basis_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install analysis extras: pip install -e '.[analysis]'") from exc
    fig,ax=plt.subplots(figsize=(9,5.5))
    for policy in policies:
        g=[r for r in summary if r['policy']==policy]
        if not g: continue
        ax.plot([r['load_basis'] for r in g],[r['acceptance_rate_mean'] for r in g],marker='o',label=policy)
    ax.set_ylabel('Workload acceptance rate');ax.set_xlabel('Offered-load definition');ax.set_ylim(-0.03,1.03)
    ax.set_title('Acceptance sensitivity to resource/load definition');ax.grid(True,alpha=.25);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(args.input_dir/'m9_acceptance_by_load_basis.png',dpi=160);plt.close(fig)
    lines=['# M9 — Load-Basis Sensitivity Report','',
      '## Purpose','',
      'M8 used the paper Table-I scalar Slice units to define cumulative offered load, while the simulator tracks compute and memory independently. This experiment asks whether that choice changes apparent saturation and baseline outcomes.','',
      '## Method','',
      '- Cluster size and arrival order are fixed for a given basis/scenario/seed.','- All jobs arrive at time zero, so this is a burst admission stress test; it is not the paper’s online-arrival protocol.','- `paper_slice_units`: stop when Table-I scalar units reach 7 per GPU.','- `compute`: stop when requested compute slices reach 7 per GPU.','- `memory`: stop when requested memory slices reach 8 per GPU.','- `max_normalized`: stop when either normalized compute or memory demand reaches 100% of its corresponding capacity.','- Same policy implementations: paper-style FF/RR, BF-BI, WF-BI, MFI.','',
      '## Results','',
      '| Load basis | Policy | Mean acceptance | Mean offered paper units | Mean compute demand | Mean memory demand | Peak compute occupancy | Peak memory occupancy |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['load_basis']} | {r['policy']} | {r['acceptance_rate_mean']:.4f} | {r['offered_paper_units_fraction_mean']:.3f} | {r['offered_compute_fraction_mean']:.3f} | {r['offered_memory_fraction_mean']:.3f} | {r['peak_arrival_compute_utilization_mean']:.3f} | {r['peak_arrival_memory_utilization_mean']:.3f} |")
    lines += ['', '## Interpretation cautions','',
      '1. This is a sensitivity experiment to expose the consequences of load-axis definitions, not a reproduction of the MFI paper.','2. The paper uses a scalar Slice column but also discusses separate compute and memory footprints; its complete source code is not available here.','3. The burst setup intentionally removes job turnover during arrivals. It is a stress test for placement/admission, not a realistic temporal model.','4. These results help diagnose a benchmark model; they do not establish a universally better scheduler.','', '## Figure','', '- `m9_acceptance_by_load_basis.png`','', '## Sources','', '- MFI paper: https://arxiv.org/abs/2511.18906','- Optimal Workload Placement on MIG: https://arxiv.org/abs/2409.06646']
    (args.input_dir/'M9_LOAD_BASIS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"Saved load-basis summaries and plots to {args.input_dir}")
if __name__=='__main__': main()
