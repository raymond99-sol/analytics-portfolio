# CV-ready research project section

Use these bullets only after you have reviewed the notebooks and can explain the methods.

## E-commerce Customer Segmentation & Revenue Strategy

**Python, SQL, scikit-learn, RFM Analysis**

- Analyzed **541,909 e-commerce transaction records** using SQL
  and Python, cleaning cancellations and missing customer identifiers to build revenue,
  order, and customer KPIs across **4,338 customers**.
- Developed an RFM-based K-means segmentation model that identified a
  **16.6% Champions segment generating
  64.9% of cleaned revenue**, and translated
  segment profiles into retention, reactivation, and second-purchase strategies;
  evaluated K=2 through K=8 and confirmed strong K=4 repeated-seed stability.

## Bank Marketing Campaign Optimization

**Python, SQL, scikit-learn, Classification**

- Developed and validated logistic-regression and random-forest pipelines on
  **41,176 bank campaign observations**, selecting the model through five-fold
  training-set PR-AUC and excluding post-call duration to prevent target leakage.
- Designed a pre-call targeting strategy whose highest-ranked 20% of holdout observations
  captured **65.9% of subscribers** at
  **3.30x baseline lift**; quantified uncertainty with 1,000 bootstrap samples and
  proposed a randomized pilot to measure incremental impact.

## Short interview explanations

### Retail project

“I wanted to turn raw transaction data into a campaign decision. I first removed
cancellations, invalid prices and quantities, and records without customer IDs. I used
SQL for core KPIs, then created recency, frequency, and monetary features in Python.
Because these variables were highly skewed, I log-transformed and standardized them
before K-means clustering. K=2 had the highest silhouette score, but I retained K=4 as
the more decision-useful framework and checked its stability across 25 random seeds. I
treated the segments as targeting hypotheses and recommended controlled tests rather than
claiming causal lift.”

### Bank project

“The business objective was to prioritize calls, so I focused on ranking metrics rather
than accuracy. I isolated the holdout before using five-fold training validation to
compare a baseline, logistic regression, and random forest. I excluded call duration
because it is only available after the call; a leakage audit showed it would inflate
ROC-AUC from 0.814 to 0.946. After calibrating the selected model, I used bootstrap
intervals to test whether top-20% lift and capture were stable. My next step would be a
later-period validation followed by a randomized pilot.”
