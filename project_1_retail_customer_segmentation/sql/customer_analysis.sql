-- Portfolio project: E-commerce customer segmentation
-- Table grain: one cleaned invoice line per row.

-- 1. Core revenue KPIs
SELECT
    ROUND(SUM(Revenue), 2) AS revenue_gbp,
    COUNT(DISTINCT InvoiceNo) AS orders,
    COUNT(DISTINCT CustomerID) AS customers,
    ROUND(SUM(Revenue) / COUNT(DISTINCT InvoiceNo), 2) AS average_order_value_gbp
FROM transactions;

-- 2. Monthly revenue and active-customer trend
SELECT
    InvoiceMonth AS month,
    ROUND(SUM(Revenue), 2) AS revenue_gbp,
    COUNT(DISTINCT InvoiceNo) AS orders,
    COUNT(DISTINCT CustomerID) AS active_customers
FROM transactions
GROUP BY InvoiceMonth
ORDER BY InvoiceMonth;

-- 3. Country performance with a minimum sample-size guardrail
SELECT
    Country AS country,
    ROUND(SUM(Revenue), 2) AS revenue_gbp,
    COUNT(DISTINCT InvoiceNo) AS orders,
    COUNT(DISTINCT CustomerID) AS customers
FROM transactions
GROUP BY Country
HAVING COUNT(DISTINCT CustomerID) >= 5
ORDER BY revenue_gbp DESC
LIMIT 10;

-- 4. Customer-level RFM inputs
SELECT
    CustomerID,
    MAX(InvoiceDate) AS last_purchase,
    COUNT(DISTINCT InvoiceNo) AS frequency,
    ROUND(SUM(Revenue), 2) AS monetary_gbp
FROM transactions
GROUP BY CustomerID;
