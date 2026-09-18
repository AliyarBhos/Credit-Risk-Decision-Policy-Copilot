DROP TABLE IF EXISTS model_scores;
DROP TABLE IF EXISTS loans;

CREATE TABLE loans (
    loan_id                 BIGINT PRIMARY KEY,

    loan_amnt               NUMERIC(12, 2)  NOT NULL,
    funded_amnt             NUMERIC(12, 2),
    term_months             SMALLINT        CHECK (term_months IN (36, 60)),
    int_rate                NUMERIC(5, 2),
    installment             NUMERIC(10, 2),
    grade                   CHAR(1)         CHECK (grade IN ('A','B','C','D','E','F','G')),
    sub_grade               VARCHAR(2),
    purpose                 VARCHAR(50),
    application_type        VARCHAR(20),

    annual_inc              NUMERIC(12, 2),
    dti                     NUMERIC(6, 2),
    emp_length_years        SMALLINT,
    home_ownership          VARCHAR(20),
    verification_status     VARCHAR(30),

    fico_range_low          SMALLINT,
    fico_range_high         SMALLINT,
    earliest_cr_line        DATE,
    open_acc                SMALLINT,
    total_acc               SMALLINT,
    mort_acc                SMALLINT,
    pub_rec                 SMALLINT,
    delinq_2yrs             SMALLINT,
    revol_bal               NUMERIC(12, 2),
    revol_util              NUMERIC(6, 2),

    addr_state              CHAR(2),
    issue_d                 DATE            NOT NULL,

    loan_status              VARCHAR(30)     NOT NULL,
    target                   SMALLINT        CHECK (target IN (0, 1)),  -- 1 = Charged Off, 0 = Fully Paid, NULL = unresolved

    loaded_at                TIMESTAMP       NOT NULL DEFAULT now()
);

CREATE INDEX idx_loans_grade ON loans (grade);
CREATE INDEX idx_loans_purpose ON loans (purpose);
CREATE INDEX idx_loans_issue_d ON loans (issue_d);
CREATE INDEX idx_loans_addr_state ON loans (addr_state);
CREATE INDEX idx_loans_target ON loans (target);


CREATE TABLE model_scores (
    score_id                BIGSERIAL PRIMARY KEY,
    loan_id                 BIGINT          NOT NULL REFERENCES loans (loan_id),

    model_version           VARCHAR(50)     NOT NULL,
    predicted_pd            NUMERIC(6, 5)   NOT NULL CHECK (predicted_pd BETWEEN 0 AND 1),
    risk_grade              CHAR(1)         NOT NULL CHECK (risk_grade IN ('A','B','C','D','E','F','G')),
    top_risk_driver         VARCHAR(100),
    top_risk_driver_value   NUMERIC(10, 5),
    auto_approve_eligible   BOOLEAN         NOT NULL DEFAULT FALSE,

    scored_at               TIMESTAMP       NOT NULL DEFAULT now()
);

CREATE INDEX idx_model_scores_loan_id ON model_scores (loan_id);
CREATE INDEX idx_model_scores_risk_grade ON model_scores (risk_grade);
CREATE INDEX idx_model_scores_scored_at ON model_scores (scored_at);


CREATE VIEW latest_model_scores AS
SELECT DISTINCT ON (loan_id) *
FROM model_scores
ORDER BY loan_id, scored_at DESC;