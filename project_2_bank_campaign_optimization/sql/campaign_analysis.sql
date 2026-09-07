-- Portfolio project: Bank marketing campaign optimization
-- Table grain: one campaign observation per customer contact record.

-- 1. Overall subscription rate
SELECT
    COUNT(*) AS contact_records,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign;

-- 2. Conversion by previous campaign outcome
SELECT
    poutcome AS previous_outcome,
    COUNT(*) AS contact_records,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY poutcome
ORDER BY conversion_rate_pct DESC;

-- 3. Job-level conversion with a minimum sample-size guardrail
SELECT
    job,
    COUNT(*) AS contact_records,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY job
HAVING COUNT(*) >= 500
ORDER BY conversion_rate_pct DESC;

-- 4. Contact-channel conversion
SELECT
    contact,
    COUNT(*) AS contact_records,
    SUM(subscribed) AS subscribers,
    ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
FROM campaign
GROUP BY contact
ORDER BY conversion_rate_pct DESC;

-- 5. Class distribution: accuracy is not an adequate model-selection metric
SELECT
    subscribed,
    COUNT(*) AS contact_records,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM campaign
GROUP BY subscribed
ORDER BY subscribed;

-- 6. Source-coded unknown values in major customer attributes
SELECT
    SUM(CASE WHEN job = 'unknown' THEN 1 ELSE 0 END) AS unknown_job,
    SUM(CASE WHEN education = 'unknown' THEN 1 ELSE 0 END) AS unknown_education,
    SUM(CASE WHEN "default" = 'unknown' THEN 1 ELSE 0 END) AS unknown_default,
    SUM(CASE WHEN housing = 'unknown' THEN 1 ELSE 0 END) AS unknown_housing,
    SUM(CASE WHEN loan = 'unknown' THEN 1 ELSE 0 END) AS unknown_loan
FROM campaign;
