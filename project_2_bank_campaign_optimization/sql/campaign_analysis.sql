-- Portfolio project: Bank marketing campaign optimization
-- Table grain: one campaign observation per customer contact record.

-- 1. Overall subscription rate
SELECT
    COUNT(*) AS customers,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign;

-- 2. Conversion by previous campaign outcome
SELECT
    poutcome AS previous_outcome,
    COUNT(*) AS customers,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY poutcome
ORDER BY conversion_rate_pct DESC;

-- 3. Job-level conversion with a minimum sample-size guardrail
SELECT
    job,
    COUNT(*) AS customers,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY job
HAVING COUNT(*) >= 500
ORDER BY conversion_rate_pct DESC;

-- 4. Contact-channel conversion
SELECT
    contact,
    COUNT(*) AS customers,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY contact
ORDER BY conversion_rate_pct DESC;
