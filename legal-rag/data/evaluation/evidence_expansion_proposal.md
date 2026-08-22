# Evidence Expansion Proposal (v6 Exploration)

## 1. Oracle Context Size Analysis
- **Baseline Context Size**: `6341.9` tokens
- **Oracle A (Supplement missing gold evidence)**: avg `+1.11` chunks, `+524.3` tokens
- **Oracle B (Keep all gold article chunks)**: avg `+1.59` chunks, `+716.8` tokens

## 2. Evidence Expansion Strategy
The forensics indicate that **Evidence Insufficiency (D)** is primarily caused by two issues:
1. **D2/D3 (60% of Class D)**: The gold article was retrieved but only some child chunks were selected, leaving behind the specific clauses/points containing the answer. This is a sibling-chunk recall issue.
2. **D7 (30% of Class D)**: The gold article's specific chunks were missed entirely by retrieval.

### Proposed Lever:
Modify the `ParentContextExpander` to expand the retrieval window to include **all child chunks of any retrieved gold parent** (or expand `neighbor_window` from 1 to 2).
This would ensure that if an article is hit, the complete article is placed in the context.

## 3. Structural Upper Bound (Ceiling)
The ceiling gain on METEOR by fixing all Class D cases is `+0.0168` on dev200.
This is **below the 1 SE threshold (~0.028)**.
Therefore, pure evidence expansion cannot yield a significant enough improvement on its own to warrant a GO verdict.

## 4. Verdict
**NO-GO**
The maximum ceiling of the proposed Evidence Expansion experiment is only +0.0168 METEOR, which is well below the 1 SE decision threshold of +0.028. Additionally, expanding the window increases average context size by ~1.2k tokens, increasing LLM generation cost without a corresponding significant gain.