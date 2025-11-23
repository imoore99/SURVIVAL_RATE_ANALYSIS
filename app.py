#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#import packages
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from lifelines import KaplanMeierFitter, NelsonAalenFitter

#import files
import structure_loan_data
import baseline_statistics
import combined_survival_metrics

## Adding stylings

st.markdown("""
<style>
/* Target the exact hamburger menu button */
button[data-testid="stExpandSidebarButton"] {
    background-color: rgba(113, 172, 172, 0.7) !important;
    color: white !important;
    border: 2px solid #ffffff !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 18px !important;
    min-width: 45px !important;
    min-height: 45px !important;
}

button[data-testid="stExpandSidebarButton"]:hover {
    background-color: #0d5aa7 !important;
    transform: scale(1.05);
}

/* Style the spans inside the button */
button[data-testid="stExpandSidebarButton"] span {
    color: white !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

## Checkbox improvements
st.markdown("""
<style>
/* Larger checkbox and font for baseline survival rate */
.stCheckbox > label {
    font-size: 18px !important;
    padding: 12px 0 !important;
}

.stCheckbox > label > div[data-testid="stMarkdownContainer"] {
    font-size: 18px !important;
}

/* Make the actual checkbox bigger */
.stCheckbox > label > div:first-child {
    transform: scale(1.3);
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Larger, more mobile-friendly multiselect controls */
.stMultiSelect > label {
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

/* Make the multiselect dropdown larger */
.stMultiSelect [data-baseweb="select"] {
    min-height: 50px !important;
    font-size: 16px !important;
}

/* Larger text inside the multiselect */
.stMultiSelect [data-baseweb="select"] > div {
    font-size: 16px !important;
    padding: 8px 12px !important;
}

/* Make the dropdown options larger when opened */
.stMultiSelect [role="listbox"] [role="option"] {
    font-size: 16px !important;
    padding: 12px 16px !important;
    min-height: 48px !important;
}

/* Better spacing for selected items */
.stMultiSelect [data-baseweb="tag"] {
    font-size: 14px !important;
    padding: 6px 10px !important;
    margin: 2px !important;
}

/* Make the X button to remove selections larger */
.stMultiSelect [data-baseweb="tag"] button {
    width: 20px !important;
    height: 20px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Change multiselect button colors from red to blue theme */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #e3f2fd !important; /* Light blue background */
    border: 1px solid #1f77b4 !important;
    color: #1f77b4 !important;
}

/* Change the X button color from red to blue */
.stMultiSelect [data-baseweb="tag"] button {
    color: rgba(113, 172, 172, 0.7)  !important;
    background-color: transparent !important;
}

.stMultiSelect [data-baseweb="tag"] button:hover {
    background-color: rgba(113, 172, 172, 0.7) !important;
    color: white !important;
}

/* Style the dropdown arrow and borders */
.stMultiSelect [data-baseweb="select"] {
    border-color: rgba(113, 172, 172, 0.7)  !important;
}

/* Selected item hover state */
.stMultiSelect [data-baseweb="tag"]:hover {
    background-color: #bbdefb !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Change checkbox color from red to blue */
.stCheckbox > label > div:first-child > div > input[type="checkbox"]:checked + div {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}

/* Change the checkmark color to white */
.stCheckbox > label > div:first-child > div > input[type="checkbox"]:checked + div::after {
    color: white !important;
}

/* Hover state for checkbox */
.stCheckbox > label > div:first-child > div:hover {
    border-color: #1f77b4 !important;
}

/* Focus state for better accessibility */
.stCheckbox > label > div:first-child > div > input[type="checkbox"]:focus + div {
    box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style the baseline statistics table headers */
.stDataFrame thead th {
    background-color: #2c3e50 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 12px 8px !important;
    text-align: center !important;
    border: 1px solid #34495e !important;
}

/* Style the row headers (index column) */
.stDataFrame tbody th {
    background-color: #2c3e50 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 12px 8px !important;
    text-align: center !important;
    border: 1px solid #34495e !important;
}

/* Keep data cells clean */
.stDataFrame tbody td {
    font-size: 16px !important;
    text-align: center !important;
    padding: 10px 8px !important;
    border: 1px solid #ddd !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Center all section headers */
.stApp h2 {
    text-align: center !important;
    margin-bottom: 1.5rem !important;
}

.stApp h3 {
    text-align: center !important;
    margin-bottom: 1rem !important;
}

/* Specifically target your section headers */
.stMarkdown h2,
.stMarkdown h3 {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style ALL dataframe headers consistently - both baseline and survival statistics */
.stDataFrame thead th {
    background-color: #2c3e50 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 12px 8px !important;
    text-align: center !important;
    border: 1px solid #34495e !important;
}

/* Style row headers (index column) for consistency */
.stDataFrame tbody th {
    background-color: #2c3e50 !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 12px 8px !important;
    text-align: center !important;
    border: 1px solid #34495e !important;
}

/* Keep data cells clean and readable */
.stDataFrame tbody td {
    font-size: 16px !important;
    text-align: center !important;
    padding: 10px 8px !important;
    border: 1px solid #ddd !important;
}
</style>
""", unsafe_allow_html=True)

# Page Configuration
st.set_page_config(page_title="Credit Portfolio Survival Analysis", 
                   page_icon=None, 
                   initial_sidebar_state='auto',
                   layout='centered')  # Centers on browser

# In[ ]:
# Sidebar Components
rate_period = st.sidebar.multiselect("Select Rate Period", ["Post-Fed Rate Increase", "Pre-Fed Rate Increase"], default=["Post-Fed Rate Increase"])
score_tier = st.sidebar.multiselect("Select Score Tier", ['Prime', 'Super-Prime', 'Subprime', 'Near-Prime'], default=['Prime', 'Super-Prime', 'Subprime', 'Near-Prime'])
baseline = st.sidebar.checkbox(label="Include Baseline Survival Rate", label_visibility="visible", width="content")

# In[ ]:
# Title and Text Components
st.title("Credit Portfolio Survival Analysis")
st.header("Kaplan-Meier Survival Curves Applied to Unsecured Loan Portfolio")

site_url = "https://cdn.prod.website-files.com/688125a82bfc6e536cc30914/689c191d8c33833818dbe635_SURVIVAL_RATE_ANALYSIS.pdf"
git_url = "https://github.com/imoore99/SURVIVAL_RATE_ANALYSIS"
st.markdown("This application allows users to explore the survival rates of unsecured loan portfolio using Kaplan-Meier survival curves. Adjust the parameters in the sidebar to see how different factors affect survival rates. " \
"The rate period is defined as the time before or after the Federal Reserve's interest rate changes. The rate tier is based on the borrower's credit score: " \
"**Super-Prime** (730+), **Prime** (650-729), **Near-Prime** (600-649), **Subprime** (below 599)." \
    " To learn more about the methodology and analysis, please refer to the detailed report [here](%s). The source code is available in this [GitHub repository](%s)." % (site_url, git_url))
st.divider(width="stretch")

# In[ ]:
# Load Data
loan_data = structure_loan_data.structure_loan_data('loan_data.csv')
# start of Fed rate increase simulation
mid_date = datetime(2022, 4, 1)
loan_data['rate_status'] = loan_data['open_date'].apply(lambda x: 'Pre-Fed Rate Increase' if x < mid_date else 'Post-Fed Rate Increase')
survival_data = structure_loan_data.prepare_survival_data(loan_data, '01-31-2025')
survival_data['risk_rate_segment'] = survival_data.apply(lambda row: f"{row['score_bucket']}, {row['rate_status']}", axis=1)

# In[ ]:
#Import kmf and naf objects
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Initialize Kaplan-Meier Fitter
    kmf = KaplanMeierFitter()
        
    # Fit the survival curve
    kmf.fit(survival_data['duration_months'], 
                survival_data['event'],
                label='Portfolio Baseline Survival Rate')

    # Initialize Nelson Aalen Fitter
    naf = NelsonAalenFitter()

    # Fit hazard curve
    naf.fit(survival_data['duration_months'], 
            survival_data['event'],
            label= 'Portfolio Baseline Hazard Rate')

# In[ ]:
#create baseline survival statistics dataframe
baseline_df = baseline_statistics.generate_survival_statistics(kmf, naf)
baseline_df = baseline_df.style\
        .set_properties(**{'color': 'black'}, **{'font-size': '24px'})
st.subheader("Baseline Survival Statistics")
st.dataframe(baseline_df)

# For loans that DO default, what's the median time?
defaulted_loans = survival_data[survival_data['event'] == 1]
median_time_to_default = defaulted_loans['duration_months'].median()
st.markdown(f"""**Median time to default (for loans that default):**
            **{median_time_to_default:.1f} months**""")

# In[ ]:
colors = [
    {"label": 'Super-Prime, Pre-Fed Rate Increase', "color": "#05409e"},
    {"label": 'Prime, Pre-Fed Rate Increase', "color": "#2470b9"},
    {"label": 'Near-Prime, Pre-Fed Rate Increase', "color": "#4599d1"},
    {"label": 'Subprime, Pre-Fed Rate Increase', "color": "#68bee8"},
    {"label": 'Super-Prime, Post-Fed Rate Increase', "color": "#d61f1f"},
    {"label": 'Prime, Post-Fed Rate Increase', "color": "#e04441"},
    {"label": 'Near-Prime, Post-Fed Rate Increase', "color": "#e76447"},
    {"label": 'Subprime, Post-Fed Rate Increase', "color": "#e58638"},
    {"label": 'Pre-Fed Rate Increase', "color": "#05409e"},
    {"label": 'Post-Fed Rate Increase', "color": "#d61f1f"},
    {"label": 'Super-Prime', "color": "#05409e"},
    {"label": 'Prime', "color": "#4599d1"},
    {"label": 'Near-Prime', "color": "#e04441"},
    {"label": 'Subprime', "color": "#e58638"}
]

fig, styled_survival_rate_summary = combined_survival_metrics.create_combined_survival_analysis(survival_data, rate_period, score_tier, colors, baseline)

st.divider(width="stretch")
st.subheader("Survival Analysis by Credit Risk Segment")
st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'displaylogo': False
})

# In[ ]:
st.divider(width="stretch")
st.subheader("Survival Analysis Statistics")
st.dataframe(styled_survival_rate_summary, hide_index=True)
