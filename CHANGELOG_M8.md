# Milestone 8 changelog

- Added a replication-oriented progressive offered-load runner for MFI's four workload distributions.
- Uses matched workload traces across First Fit, Round Robin, BF-BI, WF-BI, and MFI.
- Records 10 offered-load checkpoints and per-run metadata.
- Added summary means/standard deviations, paired acceptance deltas vs MFI, bootstrap intervals, and visualization outputs.
- Added explicit MFI model/protocol fidelity audit and paper-scale vs quick-run commands.
- Added tests for progressive workload construction and baseline execution.
- Did not add a new scheduler; the focus is baseline and protocol credibility.
