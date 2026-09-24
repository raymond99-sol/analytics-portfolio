# P1 Bank Marketing Project Handoff

## 1. Project Identity

- Research Project ID: `P1`
- Repository directory: `project_2_bank_campaign_optimization`
- Final paper title: *Decision Time Information and the Validity of Response Models for Constrained Marketing*

## 2. Research Question

How does the information available at different decision times affect the measured predictive validity and constrained-targeting value of marketing response models?

## 3. Data and Final Sample

- Source: UCI Bank Marketing, `bank-additional-full`
- Original contact records: 41,188
- Exact duplicates removed: 12
- Final observations: 41,176
- Observed subscriptions: 4,639 (11.3%)

## 4. Primary Empirical Design

- Primary information set: strict planning at the campaign-list decision point
- Predictors (10): `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `pdays`, `previous`, `poutcome`
- Split: fixed stratified 80/20 development split (32,940 training; 8,236 holdout)
- Selection: training-only model and calibration selection
- Final model: Random Forest with isotonic calibration, both selected using training-only procedures
- Evaluation status: fixed, previously inspected development holdout; not pristine prospective validation

## 5. Headline Results

- PR-AUC: 0.339
- ROC-AUC: 0.706
- Brier score: 0.087
- Top-20% capture: 45.5%
- Top-20% lift: 2.27x

## 6. Central Paired Information-Set Results

Differences are comparison minus reference, using identical holdout indices within each of 1,000 paired bootstrap resamples.

- Strict planning -> Planning + macro: Delta PR-AUC +0.115 [0.094, 0.138]; Delta Top-20% capture +19.8 pp [17.1, 22.6]
- Strict planning -> Operational pre-contact: Delta PR-AUC +0.084 [0.066, 0.099]; Delta Top-20% capture +14.7 pp [12.1, 17.4]
- Strict planning -> Operational + macro: Delta PR-AUC +0.145 [0.124, 0.167]; Delta Top-20% capture +19.6 pp [17.2, 22.7]
- Operational + macro -> Invalid duration specification: Delta PR-AUC +0.167 [0.149, 0.185]; Delta Top-20% capture +22.4 pp [19.5, 25.1]

## 7. Central Contribution

- Performance claims must correspond to the information actually available at the decision time.
- In this study, information-set specification produces materially larger measured differences than switching among the competent algorithms examined.
- The post-contact duration model is a leakage diagnostic, not a deployable targeting model.
- Specification differences are descriptive contrasts between complete information sets, not causal effects of individual variables.

## 8. Major Limitations

- Historical, widely reused dataset from one Portuguese bank
- No stable customer identifier
- No complete row-level timestamps
- Random split cannot guarantee customer-grouped or temporal separation
- Development holdout was previously inspected
- No prospective or external validation
- Response propensity is not uplift or a causal treatment effect
- No contact-cost or customer-value data; no monetary ROI claim

## 9. Final Artifacts

- `paper/AMS_2027_Blind_Manuscript.docx`
- `paper/AMS_2027_Blind_Manuscript.pdf`
- `paper/Rong_Zhao_Writing_Sample.docx`
- `paper/Rong_Zhao_Writing_Sample.pdf`
- `paper/REFERENCE_AUDIT.md`
- `paper/AMS_SUBMISSION_CHECKLIST.md`
- `paper/manuscript_source.md`
- `paper/build_manuscripts.py`

## 10. Final Repository State

- Final empirical commit: `984ef58303e2181e51dabcfc2e46fa10dfa4d66d`
- Final manuscript snapshot commit: `ca6d83e723cb05b93127fb526f32ec532f5867f0`
- Handoff creation commit: `87990c82f807f9febd402a81d4f347a6dac39966`
- This commit adds only `PROJECT_HANDOFF.md`; the manuscript snapshot remains `ca6d83e723cb05b93127fb526f32ec532f5867f0`.
- Current status: manuscript complete; ready for AMS 2027 submission.

## 11. Current Baseline and Revision Rule

- The current manuscript and empirical results are the stable baseline.
- Do not alter the project merely to chase higher predictive performance or add unnecessary complexity.
- Future revisions are allowed when motivated by verified errors, faculty or advisor feedback, reviewer comments, submission requirements, or an explicitly approved new research decision.
- After any substantive revision, update this handoff so it reflects the new authoritative project state.
