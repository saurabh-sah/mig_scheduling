# My Notes — Milestone 5

Fill this in yourself after running the diagnostics.

## 1. Experimental configuration
- Command used:
- Number of jobs / GPUs:
- Seeds:
- Scenarios:
- Fixed-lookahead horizon / rollouts:
- Adaptive rollout limits:

## 2. Baseline sanity
- Does MFI run without invalid placements?
- Did all policies see identical jobs for a given scenario/seed?
- Any unexpected results?

## 3. Adaptive behavior
- Which fallback reason was most common?
- What fraction of arrivals invoked lookahead?
- What fraction of adaptive choices differed from MFI?
- Did lookahead stop due to confidence or exhaust its budget?

## 4. Acceptance and cost
- Which scenarios, if any, had a positive paired acceptance difference versus MFI?
- How variable were the paired differences across seeds?
- How much more decision time did fixed/adaptive lookahead use?
- Did adaptive save computation compared with fixed lookahead?

## 5. Regime shift
- How did diagnostics change after demand shifted?
- Did forecast reliability fall?
- Did it return quickly? What evidence supports that?

## 6. Interpretation
- What do the data support?
- What do they not support?
- Which result surprised me?
- What should be tested next?

## 7. Go/no-go
- Continue adaptive lookahead / revise / pivot:
- Evidence:
- What extra seed count or experiment is needed before claiming anything:
