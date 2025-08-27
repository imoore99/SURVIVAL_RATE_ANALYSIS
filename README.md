## Survival Rate Analysis - Personal Loan Default Risk

#### Time-to-Event Analytics for Credit Risk Management

##### PROJECT OVERVIEW:

- Purpose: To analyze default risk patterns and loan performance characteristics within an unsecured personal loan portfolio.
- Objective: Identify critical risk inflection points and borrower stress patterns across credit segments and rate environments.

##### BUSINESS PROBLEM:

Traditional credit risk models focus on binary default outcomes but fail to capture the critical timing dimension of when defaults occur. Understanding default timing patterns is essential for optimizing portfolio management, setting appropriate loss provisions, and implementing proactive intervention strategies. This analysis addresses the need for time-aware credit risk assessment that can identify vulnerable periods in the loan lifecycle and inform strategic portfolio management decisions.

##### DATA SOURCES:

- Primary: 4,591 unsecured personal loans spanning 2021-2025
- Secondary: Interest rate environment data and economic indicators
- Features: Credit scores, loan characteristics, and borrower demographics
- Frequency: Monthly loan performance data with censoring adjustments

##### KEY FINDINGS & RESULTS:

- Critical Inflection Point: 24-30 month period identified as peak default risk window
- Rate Environment Impact: Elevated rates accelerate default timing from 30+ months to 18-24 months
- Credit Risk Spread: 14.1 percentage point difference between highest-risk (17.3% default rate) and lowest-risk (3.2% default rate) borrowers
- Risk Segmentation: Clear differentiation across credit score buckets with actionable thresholds
- Strategic Framework: Three-pillar risk management approach including dynamic concentration limits, risk-adjusted underwriting, and early warning systems

##### TECHNOLOGIES USED:

Core Stack:

- Python - Primary analysis environment
- lifelines - Kaplan-Meier survival estimation and Nelson-Aalen hazard modeling
- pandas/numpy - Data manipulation and numerical computing
- scikit-learn - Credit risk segmentation and preprocessing
- matplotlib/seaborn - Survival curves and risk heatmap visualizations

Statistical Libraries:

- scipy/statsmodels - Statistical testing and censoring adjustments
- plotly - Interactive survival curve analysis
- sklearn.metrics - Model evaluation and performance metrics

Development Tools:

- Jupyter - Interactive analysis environment
- Git - Version control and collaboration

LIVE PROJECT:
- View Full Analysis & Visualizations → https://ian-moore-analytics.webflow.io/project/garch
- Explore the complete project with interactive Kaplan-Meier curves, credit risk heatmaps, and comprehensive survival analysis insights on my portfolio site.

CONTACT:

Ian Moore - Business Intelligence, Credit Risk and Financial Analytics Leader

📧 EMAIL: ian.moore@hey.com

💼 LinkedIn: https://www.linkedin.com/in/ian-moore-analytics/

🌐 Portfolio: https://www.ianmooreanalytics.com

This project demonstrates advanced quantitative risk analysis techniques for institutional portfolio management and strategic investment decision-making.