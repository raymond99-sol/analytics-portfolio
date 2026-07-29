# Resume-ready project section

Use these bullets only after you have reviewed the notebooks and can explain the methods.

## E-commerce Customer Segmentation & Revenue Strategy

**Python, SQL, scikit-learn, RFM Analysis**

- Analyzed **541,909 e-commerce transaction records** using SQL
  and Python, cleaning cancellations and missing customer identifiers to build revenue,
  order, and customer KPIs across **4,338 customers**.
- Developed an RFM-based K-means segmentation model that identified a
  **16.6% Champions segment generating
  64.9% of cleaned revenue**, and translated
  segment profiles into retention, reactivation, and second-purchase strategies.

## Bank Marketing Campaign Optimization

**Python, SQL, scikit-learn, Classification**

- Built and compared logistic-regression and random-forest pipelines on
  **41,176 bank campaign observations**, excluding post-call
  duration to prevent target leakage and evaluating performance with ROC-AUC and PR-AUC.
- Designed a pre-call targeting strategy whose highest-ranked 20% of holdout customers
  captured **65.9% of subscribers** at
  **3.30x baseline lift**, with a randomized-pilot plan
  to measure incremental impact.

## Short interview explanations

### Retail project

“I wanted to turn raw transaction data into a campaign decision. I first removed
cancellations, invalid prices and quantities, and records without customer IDs. I used
SQL for core KPIs, then created recency, frequency, and monetary features in Python.
Because these variables were highly skewed, I log-transformed and standardized them
before K-means clustering. I treated the segments as targeting hypotheses and recommended
controlled tests rather than claiming causal lift.”

### Bank project

“The business objective was to prioritize calls, so I focused on ranking metrics rather
than accuracy. I excluded call duration because it is only available after the call and
would leak the outcome. I compared two models, evaluated PR-AUC because the target was
imbalanced, and measured how many subscribers appeared in the top 20% of scores. My next
step would be a temporal validation followed by a randomized pilot.”
