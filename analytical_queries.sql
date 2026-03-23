/*
 SQL Analytics Toolkit — Operational KPI & SLA Reporting
 =========================================================
 Production analytical queries demonstrating:
 - CTEs & recursive CTEs
 - Window functions (RANK, NTILE, LAG/LEAD, rolling averages)
 - YoY & MoM growth calculations
 - SLA compliance tracking
 - Cohort analysis
 - Pareto analysis (80/20 rule)
 - Ad-hoc investigation patterns

 Compatible with: PostgreSQL, Redshift, Snowflake
 Author: Thriksha Giriraju
*/


-- ─────────────────────────────────────────────
-- 1. SLA COMPLIANCE DASHBOARD QUERY
-- Rolling 30-day SLA breach rate with trend
-- ─────────────────────────────────────────────
WITH daily_sla AS (
    SELECT
        DATE(created_at) AS report_date,
        COUNT(*) AS total_tickets,
        SUM(CASE WHEN resolved_at <= sla_deadline THEN 1 ELSE 0 END) AS met_sla,
        SUM(CASE WHEN resolved_at > sla_deadline THEN 1 ELSE 0 END) AS breached_sla
    FROM incidents
    GROUP BY DATE(created_at)
),
rolling_metrics AS (
    SELECT
        report_date,
        total_tickets,
        breached_sla,
        ROUND(breached_sla::DECIMAL / NULLIF(total_tickets, 0) * 100, 2) AS daily_breach_pct,
        AVG(breached_sla::DECIMAL / NULLIF(total_tickets, 0) * 100)
            OVER (ORDER BY report_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
            AS rolling_30d_breach_pct,
        LAG(breached_sla::DECIMAL / NULLIF(total_tickets, 0) * 100, 7)
            OVER (ORDER BY report_date) AS breach_pct_7d_ago
    FROM daily_sla
)
SELECT
    report_date,
    total_tickets,
    breached_sla,
    daily_breach_pct,
    ROUND(rolling_30d_breach_pct, 2) AS rolling_30d_breach_pct,
    ROUND(daily_breach_pct - COALESCE(breach_pct_7d_ago, daily_breach_pct), 2) AS wow_change
FROM rolling_metrics
ORDER BY report_date DESC;


-- ─────────────────────────────────────────────
-- 2. RESOLUTION TIME ANALYSIS BY PRIORITY
-- Percentile analysis with NTILE bucketing
-- ─────────────────────────────────────────────
WITH resolution_stats AS (
    SELECT
        priority,
        EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600.0 AS resolution_hrs,
        NTILE(4) OVER (PARTITION BY priority
                        ORDER BY EXTRACT(EPOCH FROM (resolved_at - created_at))) AS quartile
    FROM incidents
    WHERE resolved_at IS NOT NULL
)
SELECT
    priority,
    COUNT(*) AS ticket_count,
    ROUND(AVG(resolution_hrs)::NUMERIC, 2) AS avg_resolution_hrs,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY resolution_hrs)::NUMERIC, 2) AS median_hrs,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY resolution_hrs)::NUMERIC, 2) AS p90_hrs,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY resolution_hrs)::NUMERIC, 2) AS p95_hrs,
    ROUND(MIN(resolution_hrs)::NUMERIC, 2) AS min_hrs,
    ROUND(MAX(resolution_hrs)::NUMERIC, 2) AS max_hrs
FROM resolution_stats
GROUP BY priority
ORDER BY avg_resolution_hrs DESC;


-- ─────────────────────────────────────────────
-- 3. MONTH-OVER-MONTH & YEAR-OVER-YEAR GROWTH
-- Revenue trend analysis with growth rates
-- ─────────────────────────────────────────────
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', transaction_date) AS month,
        COUNT(DISTINCT customer_id) AS active_customers,
        SUM(amount) AS total_revenue,
        AVG(amount) AS avg_transaction
    FROM transactions
    GROUP BY DATE_TRUNC('month', transaction_date)
)
SELECT
    month,
    active_customers,
    ROUND(total_revenue, 2) AS total_revenue,
    ROUND(avg_transaction, 2) AS avg_transaction,
    -- MoM growth
    ROUND((total_revenue - LAG(total_revenue) OVER (ORDER BY month))
          / NULLIF(LAG(total_revenue) OVER (ORDER BY month), 0) * 100, 2) AS mom_growth_pct,
    -- YoY growth
    ROUND((total_revenue - LAG(total_revenue, 12) OVER (ORDER BY month))
          / NULLIF(LAG(total_revenue, 12) OVER (ORDER BY month), 0) * 100, 2) AS yoy_growth_pct,
    -- 3-month rolling average
    ROUND(AVG(total_revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
        AS rolling_3m_avg
FROM monthly_revenue
ORDER BY month DESC;


-- ─────────────────────────────────────────────
-- 4. CUSTOMER COHORT RETENTION ANALYSIS
-- Track retention by signup month
-- ─────────────────────────────────────────────
WITH customer_cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', first_purchase_date) AS cohort_month
    FROM customers
),
activity AS (
    SELECT
        t.customer_id,
        c.cohort_month,
        DATE_TRUNC('month', t.transaction_date) AS activity_month,
        (EXTRACT(YEAR FROM t.transaction_date) - EXTRACT(YEAR FROM c.cohort_month)) * 12
        + EXTRACT(MONTH FROM t.transaction_date) - EXTRACT(MONTH FROM c.cohort_month)
            AS months_since_signup
    FROM transactions t
    JOIN customer_cohorts c ON t.customer_id = c.customer_id
)
SELECT
    cohort_month,
    months_since_signup,
    COUNT(DISTINCT customer_id) AS active_customers,
    ROUND(COUNT(DISTINCT customer_id)::DECIMAL /
          FIRST_VALUE(COUNT(DISTINCT customer_id))
              OVER (PARTITION BY cohort_month ORDER BY months_since_signup) * 100, 1)
        AS retention_pct
FROM activity
WHERE months_since_signup BETWEEN 0 AND 12
GROUP BY cohort_month, months_since_signup
ORDER BY cohort_month, months_since_signup;


-- ─────────────────────────────────────────────
-- 5. PARETO ANALYSIS (80/20 RULE)
-- Which 20% of categories cause 80% of issues?
-- ─────────────────────────────────────────────
WITH category_counts AS (
    SELECT
        category,
        COUNT(*) AS incident_count,
        SUM(COUNT(*)) OVER () AS total_incidents
    FROM incidents
    GROUP BY category
),
ranked AS (
    SELECT
        category,
        incident_count,
        ROUND(incident_count::DECIMAL / total_incidents * 100, 2) AS pct_of_total,
        SUM(incident_count) OVER (ORDER BY incident_count DESC) AS cumulative_count,
        ROUND(SUM(incident_count) OVER (ORDER BY incident_count DESC)::DECIMAL
              / total_incidents * 100, 2) AS cumulative_pct,
        RANK() OVER (ORDER BY incident_count DESC) AS rank_position
    FROM category_counts
)
SELECT
    category,
    incident_count,
    pct_of_total,
    cumulative_pct,
    CASE WHEN cumulative_pct <= 80 THEN 'TOP 80%' ELSE 'REMAINING' END AS pareto_group
FROM ranked
ORDER BY incident_count DESC;


-- ─────────────────────────────────────────────
-- 6. AGENT PERFORMANCE RANKING
-- Window function-based performance scoring
-- ─────────────────────────────────────────────
WITH agent_metrics AS (
    SELECT
        agent_id,
        agent_name,
        COUNT(*) AS tickets_handled,
        ROUND(AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600.0)::NUMERIC, 2)
            AS avg_resolution_hrs,
        ROUND(SUM(CASE WHEN resolved_at <= sla_deadline THEN 1 ELSE 0 END)::DECIMAL
              / COUNT(*) * 100, 1) AS sla_compliance_pct,
        ROUND(AVG(satisfaction_score)::NUMERIC, 2) AS avg_csat
    FROM incidents
    WHERE resolved_at IS NOT NULL
    GROUP BY agent_id, agent_name
    HAVING COUNT(*) >= 20  -- minimum ticket threshold
)
SELECT
    agent_name,
    tickets_handled,
    avg_resolution_hrs,
    sla_compliance_pct,
    avg_csat,
    RANK() OVER (ORDER BY sla_compliance_pct DESC) AS sla_rank,
    RANK() OVER (ORDER BY avg_csat DESC) AS csat_rank,
    NTILE(4) OVER (ORDER BY sla_compliance_pct DESC) AS performance_quartile
FROM agent_metrics
ORDER BY sla_compliance_pct DESC;


-- ─────────────────────────────────────────────
-- 7. ANOMALY DETECTION — TICKETS OUTSIDE NORMAL RANGE
-- Flag days with unusual ticket volume
-- ─────────────────────────────────────────────
WITH daily_volume AS (
    SELECT
        DATE(created_at) AS report_date,
        COUNT(*) AS ticket_count
    FROM incidents
    GROUP BY DATE(created_at)
),
stats AS (
    SELECT
        report_date,
        ticket_count,
        AVG(ticket_count) OVER (ORDER BY report_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
            AS rolling_avg,
        STDDEV(ticket_count) OVER (ORDER BY report_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
            AS rolling_std
    FROM daily_volume
)
SELECT
    report_date,
    ticket_count,
    ROUND(rolling_avg::NUMERIC, 1) AS rolling_30d_avg,
    ROUND(rolling_std::NUMERIC, 1) AS rolling_30d_std,
    CASE
        WHEN ticket_count > rolling_avg + 2 * rolling_std THEN 'HIGH ANOMALY'
        WHEN ticket_count < rolling_avg - 2 * rolling_std THEN 'LOW ANOMALY'
        ELSE 'NORMAL'
    END AS status
FROM stats
WHERE ticket_count > rolling_avg + 2 * rolling_std
   OR ticket_count < rolling_avg - 2 * rolling_std
ORDER BY report_date DESC;
