"""Create plots from a completed M12 experiment directory."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

POLICY_ORDER = [
    "paper_ff", "paper_rr", "paper_bf_bi", "paper_wf_bi",
    "global_feasible_bf_bi", "global_feasible_wf_bi",
    "mfi_all_starts", "mfi_free_anchor_score",
]
POLICY_LABELS = {
    "paper_ff": "Paper-path FF", "paper_rr": "Paper-path RR",
    "paper_bf_bi": "Paper-path BF-BI", "paper_wf_bi": "Paper-path WF-BI",
    "global_feasible_bf_bi": "Global-feasible BF-BI",
    "global_feasible_wf_bi": "Global-feasible WF-BI",
    "mfi_all_starts": "MFI: all starts", "mfi_free_anchor_score": "MFI: free-anchor score",
}
SCENARIOS = ["uniform", "skew_small", "skew_big", "bimodal"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    args=ap.parse_args()
    out=args.results
    run_rows=read_csv(out/"m12_raw_runs.csv")
    cp_rows=read_csv(out/"m12_checkpoint_rows.csv")
    # Figure 1: acceptance at 85% offered demand, matching the paper's heavy-load snapshot.
    target_cp=min({float(r['checkpoint_requested']) for r in cp_rows}, key=lambda x: abs(x-0.85))
    rows=[r for r in cp_rows if abs(float(r['checkpoint_requested'])-target_cp)<1e-8]
    meanvals={}
    for pol in POLICY_ORDER:
        meanvals[pol]={}
        for scen in SCENARIOS:
            vals=[float(r['acceptance_rate_so_far']) for r in rows if r['policy']==pol and r['scenario']==scen]
            meanvals[pol][scen]=sum(vals)/len(vals) if vals else 0
    fig, ax=plt.subplots(figsize=(12,6.5))
    width=.1
    positions=list(range(len(SCENARIOS)))
    for i,pol in enumerate(POLICY_ORDER):
        offset=(i-(len(POLICY_ORDER)-1)/2)*width
        ax.bar([x+offset for x in positions], [meanvals[pol][s]*100 for s in SCENARIOS], width=width, label=POLICY_LABELS[pol])
    ax.set_xticks(positions, SCENARIOS)
    ax.set_ylabel("Acceptance by offered-load checkpoint (%)")
    ax.set_title(f"M12: Acceptance at {target_cp:.0%} requested scalar slice units")
    ax.set_ylim(0,105)
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"m12_acceptance_at_85pct.png", dpi=180)
    fig.savefig(out/"m12_acceptance_at_85pct.svg")
    plt.close(fig)

    # Figure 2: paired means across scenario/seed for full traces.
    by_policy={}
    for pol in POLICY_ORDER:
        vals=[float(r['acceptance_rate']) for r in run_rows if r['policy']==pol]
        by_policy[pol]=sum(vals)/len(vals) if vals else 0
    fig,ax=plt.subplots(figsize=(11,6))
    ax.bar([POLICY_LABELS[p] for p in POLICY_ORDER], [by_policy[p]*100 for p in POLICY_ORDER])
    ax.set_ylabel("Full-trace acceptance (%)")
    ax.set_title("M12: Full-trace acceptance across scenarios and repetitions")
    ax.set_ylim(0,105)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out/"m12_full_trace_acceptance.png", dpi=180)
    fig.savefig(out/"m12_full_trace_acceptance.svg")
    plt.close(fig)

    print(f"Wrote plots to {out}")

if __name__ == "__main__":
    main()
