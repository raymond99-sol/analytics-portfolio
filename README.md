# Rong Zhao - Research-Oriented Analytics Portfolio

Two reproducible projects at the intersection of business analytics, quantitative
marketing, and information systems:

1. **E-commerce Customer Segmentation & Revenue Strategy**  
   SQL KPIs, RFM analysis, K-means clustering, and campaign recommendations.
2. **Bank Marketing Campaign Optimization**  
   SQL conversion analysis, classification pipelines, leakage prevention, lift, and
   targeting strategy.

Both projects use public UCI datasets under CC BY 4.0 and include executed notebooks,
standalone SQL, saved results, charts, research limitations, resume bullets, and a
validation report. Raw source data is intentionally excluded from GitHub and can be
downloaded with the included setup script.

## Start here

1. Read [`PROJECTS_FOR_RESUME.md`](PROJECTS_FOR_RESUME.md).
2. Review the two executed notebooks and their project-specific README files.
3. Practice explaining the business question, data cleaning, method choice, key result,
   limitation, and recommended next step.
4. Download the source data and rerun the notebooks locally when you want to reproduce
   the analysis.

## Reproduce locally

```bash
git clone https://github.com/raymond99-sol/analytics-portfolio.git
cd analytics-portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r project_1_retail_customer_segmentation/requirements.txt
python download_data.py
jupyter lab
```

Open either notebook. Each notebook can resolve its project directory when Jupyter is
launched from the repository root or from the individual project folder.

## Project highlights

- **Retail segmentation:** 541,909 raw invoice lines, 4,338 customers, and an RFM-based
  Champions segment representing 16.6% of customers and 64.9% of cleaned revenue;
  K=2 through K=8 diagnostics and repeated-seed stability make the K=4 choice auditable.
- **Bank campaign optimization:** training-only five-fold model selection, probability
  calibration, leakage auditing, and 1,000-sample bootstrap inference; the top 20% of
  holdout scores captured 65.9% of subscribers at 3.30x baseline lift.

## Portfolio integrity

These are completed portfolio projects, not employment experience. Put them in a
**PROJECTS** section and be prepared to explain the work. Do not claim that the
recommendations were deployed or that they caused business results.
