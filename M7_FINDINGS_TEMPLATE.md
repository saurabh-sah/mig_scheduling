# M7 Findings Template

Fill this in after running the default experiment. Do not leave a hypothesis as if it were a result.

## Reproducibility
- Test result:
- Baseline validation result:
- Number of matched traces completed:
- Any errors or skipped runs:

## Primary result
- Best fixed lookahead horizon by acceptance:
- Paired delta vs MFI:
- Stratified bootstrap 95% interval:
- Runtime multiplier vs MFI:
- Is the interval compatible with zero?

## Decision diagnostics
- How often does fixed lookahead change the MFI choice?
- How often does adaptive lookahead invoke search?
- Which fallback reason dominates?
- Does a larger top-two estimated-value gap predict realized trace benefit?
- Is the selected candidate's estimated advantage larger than its rollout variability?

## Scenario dependence
- Uniform:
- Skew-small:
- Skew-big:
- Bimodal:
- Small-to-big shift:
- Big-to-small shift:

## Go/no-go
- Continue / modify / pivot:
- Evidence supporting the decision:
- Limitations still unresolved:
- Next experiment chosen before looking at its results:
