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

# Page Configuration
st.set_page_config(page_title="Credit Portfolio Survival Analysis", 
                   page_icon=None, 
                   #layout='centered', 
                   initial_sidebar_state='auto',
                   layout="wide")  # Uses full browser width

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

st.markdown("This application allows users to explore the survival rates of unsecured loan portfolios using Kaplan-Meier survival curves. Adjust the parameters in the sidebar to see how different factors affect survival rates. View a detailed report for a comprehensive analysis [here](%s). The source code is available in this [github repository](%s)." % (site_url, git_url))
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
st.subheader("Baseline Survival Statistics")
st.dataframe(baseline_df)

# For loans that DO default, what's the median time?
defaulted_loans = survival_data[survival_data['event'] == 1]
median_time_to_default = defaulted_loans['duration_months'].median()
st.markdown(f"**Median time to default (for loans that default): {median_time_to_default:.1f} months**")

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
st.pyplot(fig)

# In[ ]:
st.divider(width="stretch")
st.subheader("Survival Analysis Statistics")
st.dataframe(styled_survival_rate_summary)
