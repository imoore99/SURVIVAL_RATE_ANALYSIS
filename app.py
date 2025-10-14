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
st.header("Kaplan-Meier Survival Curvies Applied to Unsecured Loan Portfolio")
st.text("This application allows users to explore the survival rates of unsecured loan portfolios using Kaplan-Meier survival curves. Adjust the parameters in the sidebar to see how different factors affect survival rates.")
st.divider(width="stretch")
# In[ ]:
#Import functions to structure and prepare data


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
#generate baseline surival statistics

# In[ ]:
def generate_survival_statistics(kmf, naf):
    time_points = [6, 12, 18, 24, 30, 36]
    survival_col = []
    default_prob_col = []
    cum_hazard_col = []
    for months in time_points:
            #generate survival prob for each time point
            survival_prob = kmf.survival_function_at_times(months).values[0]
            survival_col.append(
                        round(survival_prob*100, 2).astype(str) +'%'
            )
            #generate default prob for each time point
            default_prob = 1 - survival_prob
            default_prob_col.append(
                    round(default_prob*100, 2).astype(str) +'%'
            )
            #generate cumulative hazard for each time point
            cum_hazard = naf.cumulative_hazard_at_times(months).values[0]
            cum_hazard_col.append(
                    round(cum_hazard*100, 2).astype(str) +'%'
            )
    baseline_stats = pd.DataFrame({
        'Months': time_points,
        'Survival Probability': survival_col,
        'Default Probability': default_prob_col,
        'Cumulative Hazard': cum_hazard_col
    })
    baseline_stats.index = baseline_stats['Months']
    baseline_stats.drop(columns=['Months'], inplace=True)
    return baseline_stats

baseline_df = generate_survival_statistics(kmf, naf)

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

def create_combined_survival_analysis(survival_data, rate_period, score_tier, colors, baseline=True):
    #create table segments for dataframe
    risk_segment_col = []
    default_rate_col = []
    median_time_to_default_col = []
    survival_12mo_col = []
    survival_24mo_col = []
    survival_36mo_col = []
    default_size_col = []
    sample_size_col = []

    fig = plt.figure(figsize=(16, 10))

    def kmf_baseline(survival_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Initialize Kaplan-Meier Fitter
            kmf = KaplanMeierFitter()
                    
            # Fit the survival curve
            kmf.fit(survival_data['duration_months'], 
                            survival_data['event'],
                            label='Portfolio Baseline Survival Rate')
        return kmf

    kmf = kmf_baseline(survival_data)
    if baseline == True:
        baseline = kmf.plot_survival_function(ci_alpha = 0.0, color='black', linewidth=2, linestyle = '--', label='Baseline Survival Rate')
    if baseline == False:
        baseline = plt.subplot(1,1,1) # Create empty subplot for custom plotting    

    if len(score_tier) == 0:
        for i, rate in enumerate(rate_period):
            segment = f"{rate}"
            # Filter data for this segment
            segment_data = survival_data[survival_data['rate_status'] == segment]
            for i in colors:
                    if i['label'] == segment:
                        color = i['color']

            # Calculate basic metrics
            total_loans = len(segment_data)
            defaults = segment_data['event'].sum()
            default_rate = (defaults / total_loans) * 100
                        
            # Fit Kaplan-Meier for this segment
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=segment_data['duration_months'],
                event_observed=segment_data['event'],
                label=f'{segment} ({defaults} defaults | {round(default_rate, 1)}% default rate)'
                )
                        
            # Plot survival curve
            plot_objects = kmf.plot_survival_function(
                    ax=baseline,
                    color=color, 
                    linewidth=3,
                    alpha=0.8,
                    ci_alpha=0.1  # Light confidence interval
                    )
            
    elif len(rate_period) == 0:
        for i, score in enumerate(score_tier):
            segment = f"{score}"
            # Filter data for this segment
            segment_data = survival_data[survival_data['score_bucket'] == segment]
            for i in colors:
                    if i['label'] == segment:
                        color = i['color']

            # Calculate basic metrics
            total_loans = len(segment_data)
            defaults = segment_data['event'].sum()
            default_rate = (defaults / total_loans) * 100

            risk_segment_col.append(segment)
            default_rate_col.append(round(default_rate, 1))
            median_time_to_default_col.append(segment_data['duration_months'].median())
            default_size_col.append(defaults)
            sample_size_col.append(total_loans)
                        
            # Fit Kaplan-Meier for this segment
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=segment_data['duration_months'],
                event_observed=segment_data['event'],
                label=f'{segment} ({defaults} defaults | {round(default_rate, 1)}% default rate)'
                )
            survival_prob_12 = kmf.survival_function_at_times(12).values[0]
            survival_12mo_col.append(round(survival_prob_12*100, 1))

            survival_prob_24 = kmf.survival_function_at_times(24).values[0]
            survival_24mo_col.append(round(survival_prob_24*100, 1))

            survival_prob_36 = kmf.survival_function_at_times(36).values[0]
            survival_36mo_col.append(round(survival_prob_36*100, 1))
                        
            # Plot survival curve
            plot_objects = kmf.plot_survival_function(
                    ax=baseline,
                    color=color, 
                    linewidth=3,
                    alpha=0.8,
                    ci_alpha=0.1  # Light confidence interval
                    )
            
    else:
        for i, rate in enumerate(rate_period):
            for j, score in enumerate(score_tier):
                segment = f"{score}, {rate}"
                # Filter data for this segment
                segment_data = survival_data[survival_data['risk_rate_segment'] == segment]
                    
                for i in colors:
                    if i['label'] == segment:
                        color = i['color']

                # Calculate basic metrics
                total_loans = len(segment_data)
                defaults = segment_data['event'].sum()
                default_rate = (defaults / total_loans) * 100

                risk_segment_col.append(segment)
                default_rate_col.append(round(default_rate, 1))
                median_time_to_default_col.append(segment_data['duration_months'].median())
                default_size_col.append(defaults)
                sample_size_col.append(total_loans)
                        
                # Fit Kaplan-Meier for this segment
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=segment_data['duration_months'],
                    event_observed=segment_data['event'],
                    label=f'{segment} ({defaults} defaults | {round(default_rate, 1)}% default rate)'
                    )
                survival_prob_12 = kmf.survival_function_at_times(12).values[0]
                survival_12mo_col.append(round(survival_prob_12*100, 1))

                survival_prob_24 = kmf.survival_function_at_times(24).values[0]
                survival_24mo_col.append(round(survival_prob_24*100, 1))

                survival_prob_36 = kmf.survival_function_at_times(36).values[0]
                survival_36mo_col.append(round(survival_prob_36*100, 1))
                
                        
                # Plot survival curve
                plot_objects = kmf.plot_survival_function(
                    ax=baseline,
                    color=color, 
                    linewidth=3,
                    alpha=0.8,
                    ci_alpha=0.1  # Light confidence interval
                    )


            
    # Format the plot
    ax = baseline

    # Get the Figure object from the AxesSubplot
    fig = ax.get_figure()
    #plt.title('Survival Analysis by Credit Risk Segment', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Months Since Origination', fontsize=16, fontweight='bold')
    plt.ylabel('Survival Probability (No Default)', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left', fontsize=14, framealpha=0.9)
                
    # Add key milestone annotations
    plt.axvline(x=12, color='gray', linestyle='--', alpha=0.5, label='12 Month Mark')
    plt.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='24 Month Mark')
    plt.axvline(x=36, color='gray', linestyle='--', alpha=0.5, label='36 Month Mark')
                
    # Set axis limits
    plt.xlim(0, max(survival_data['duration_months']) * 1.02)
    plt.ylim(-.02, 1.02)  # Focus on the range where action happens
                
    plt.tight_layout()
     # Create summary dataframe
    survival_rate_summary = pd.DataFrame({
        'Risk Segment': risk_segment_col,
        'Default Rate (%)': default_rate_col,
        'Median Time to Default (months)': median_time_to_default_col,
        '12 Month Survival Rate (%)': survival_12mo_col,
        '24 Month Survival Rate (%)': survival_24mo_col,
        '36 Month Survival Rate (%)': survival_36mo_col,
        'Number of Defaults': default_size_col
    })
    survival_rate_summary.index = survival_rate_summary['Risk Segment']
    survival_rate_summary.drop(columns=['Risk Segment'], inplace=True)

    def highlight_rows(row):
        if row['Default Rate (%)'] > 15:  # Only high-risk segments
            return ['background-color: coral'] * len(row)  # Coral color for high-risk
        return [''] * len(row)

    styled_survival_rate_summary = survival_rate_summary.style.apply(highlight_rows, axis=1)

    return fig, styled_survival_rate_summary

fig, styled_survival_rate_summary = create_combined_survival_analysis(survival_data, rate_period, score_tier, colors, baseline)

st.divider(width="stretch")
st.subheader("Survival Analysis by Credit Risk Segment")
st.pyplot(fig)

# In[ ]:
st.divider(width="stretch")
st.subheader("Survival Analysis Statistics")
st.dataframe(styled_survival_rate_summary)
