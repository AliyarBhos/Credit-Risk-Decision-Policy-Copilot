SELECT
    grade,
    COUNT(*) AS n_loans,
    ROUND(AVG(target)::numeric, 4) AS default_rate
FROM loans
WHERE target IS NOT NULL          
GROUP BY grade
ORDER BY grade;



SELECT
    purpose,
    COUNT(*) AS n_loans,
    ROUND(AVG(target)::numeric, 4) AS default_rate,
    ROUND(AVG(loan_amnt)::numeric, 0) AS avg_loan_amnt
FROM loans
WHERE target IS NOT NULL
GROUP BY purpose
HAVING COUNT(*) >= 10               
ORDER BY default_rate DESC;



SELECT
    DATE_TRUNC('month', issue_d)::date  AS issue_month,
    COUNT(*) AS n_loans,
    ROUND(SUM(loan_amnt)::numeric, 0)  AS total_originated,
    ROUND(AVG(target)::numeric, 4) AS default_rate
FROM loans
WHERE target IS NOT NULL
GROUP BY 1
ORDER BY 1;



SELECT
    CASE
        WHEN dti < 10 THEN '0-10'
        WHEN dti < 20 THEN '10-20'
        WHEN dti < 30 THEN '20-30'
        WHEN dti < 43 THEN '30-43'
        WHEN dti < 50 THEN '43-50'
        ELSE '50+'
    END AS dti_bucket,
    COUNT(*) AS n_loans,
    ROUND(AVG(target)::numeric, 4) AS default_rate
FROM loans
WHERE target IS NOT NULL AND dti IS NOT NULL
GROUP BY 1
ORDER BY MIN(dti);



SELECT
    grade,
    COUNT(*) AS n_loans,
    ROUND(AVG(int_rate)::numeric, 2) AS avg_int_rate,
    ROUND(MIN(int_rate)::numeric, 2) AS min_int_rate,
    ROUND(MAX(int_rate)::numeric, 2) AS max_int_rate
FROM loans
GROUP BY grade
ORDER BY grade;



SELECT
    addr_state,
    COUNT(*)  AS n_loans,
    ROUND(AVG(target)::numeric, 4) AS default_rate
FROM loans
WHERE target IS NOT NULL
GROUP BY addr_state
ORDER BY n_loans DESC
LIMIT 15;



SELECT
    loan_id, loan_amnt, funded_amnt, term_months, int_rate, installment,
    grade, sub_grade, purpose, application_type,
    annual_inc, dti, emp_length_years, home_ownership, verification_status,
    fico_range_low, fico_range_high, earliest_cr_line,
    open_acc, total_acc, mort_acc, pub_rec, delinq_2yrs, revol_bal, revol_util,
    addr_state, issue_d,
    target
FROM loans
WHERE target IS NOT NULL;



SELECT
    EXTRACT(YEAR FROM issue_d)::int  AS issue_year,
    COUNT(*) AS n_loans,
    ROUND(AVG(target)::numeric, 4) AS default_rate
FROM loans
WHERE target IS NOT NULL
GROUP BY 1
ORDER BY 1;



SELECT
    s.risk_grade,
    COUNT(*) AS n_scored,
    ROUND(AVG(s.predicted_pd)::numeric, 4)  AS avg_predicted_pd,
    ROUND(AVG(l.target)::numeric, 4) AS actual_default_rate
FROM latest_model_scores s
JOIN loans l ON l.loan_id = s.loan_id
WHERE l.target IS NOT NULL
GROUP BY s.risk_grade
ORDER BY s.risk_grade;



SELECT
    l.loan_id, s.risk_grade, s.predicted_pd, s.auto_approve_eligible, s.scored_at
FROM latest_model_scores s
JOIN loans l ON l.loan_id = s.loan_id
WHERE s.risk_grade = 'G' AND s.auto_approve_eligible = TRUE;



SELECT
    l.loan_id, s.predicted_pd, s.risk_grade, l.dti, l.emp_length_years, l.annual_inc
FROM latest_model_scores s
JOIN loans l ON l.loan_id = s.loan_id
WHERE s.predicted_pd > 0.35
  AND l.dti < 30
  AND l.emp_length_years >= 2
ORDER BY s.predicted_pd DESC;