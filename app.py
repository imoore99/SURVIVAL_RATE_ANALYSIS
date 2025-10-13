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
def bucket_loan_data(loan_data, col, bucket_col, bins, labels):
    """
    Bucket loan data into categories based on rate.
    """
    # Ensure 'rate' column exists
    if 'rate' not in loan_data.columns:
        raise ValueError("Data must contain 'rate' column for bucketing.")
    
    # Create buckets based on rate
    loan_data[bucket_col] = pd.cut(
                loan_data[col], 
                bins=bins, 
                labels=labels
    )
    return loan_data

def structure_loan_data(loan_data_csv):
    """
    Structure the raw loan data into a DataFrame with specific columns.
    """
    loan_data_raw = pd.read_csv(loan_data_csv)

    
    # Initialize dataframe for loan data
    loan_data = pd.DataFrame(columns=['loan_id', 'open_date', 'credit_score', '6_month_credit_score', 'term', 'rate', 'orig_amount', 'status', 'rate_bucket', 'score_bucket', 'orig_amount_bucket', 'open_year', 'open_month', 'open_month_str', 'maturity_date'])

    loan_data['loan_id'] = loan_data_raw.index + 1000
    loan_data['open_date'] = pd.to_datetime(loan_data_raw['OPEN_DATE'])
    loan_data['credit_score'] = loan_data_raw['CREDIT_SCORE_AT_ORIG']
    loan_data['6_month_score_change'] = loan_data_raw['6_MOS_SCORE_CHG']
    loan_data['rate'] = loan_data_raw['RATE']
    loan_data['orig_amount'] = loan_data_raw['LOAN_AMOUNT']
    loan_data['status'] = loan_data_raw['STATUS']

    # Randomly assign a amtuirty date between 3 and 8 years from the open date
    loan_data['maturity_date'] = loan_data['open_date'] + pd.to_timedelta(loan_data_raw['TERM'] * 30.44, unit='D')
    loan_data['term'] = loan_data_raw['TERM']

    #Backfill other date columns
    loan_data['open_month'] = loan_data['open_date'].dt.to_period('M')
    loan_data['open_year'] = loan_data['open_date'].dt.year
    loan_data['open_month_str'] = loan_data['open_month'].astype(str)


    # Create buckets for rate
    rate_bucket_bins = [0, 10, 13, 16, 19, 22]
    rate_bucket_labels = ['Low', 'Low-Med', 'Medium', 'Med-High', 'High']
    loan_data = bucket_loan_data(loan_data, 'rate', 'rate_bucket',
                    bins=rate_bucket_bins,
                    labels=rate_bucket_labels
                    )

    # Create buckets for credit score
    score_bucket_bins=[-1, 599, 649, 729, 900]
    score_bucket_labels=['Subprime', 'Near-Prime', 'Prime', 'Super-Prime']
    loan_data = bucket_loan_data(loan_data, 'credit_score', 'score_bucket',
                    bins=score_bucket_bins,
                    labels=score_bucket_labels
                    )

    # Create buckets for orignal amount
    orig_amount_bucket_bins = [0, 5000, 10000, 20000, 30000, 50000]
    orig_amount_bucket_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    loan_data = bucket_loan_data(loan_data, 'orig_amount', 'orig_amount_bucket',
                    bins=orig_amount_bucket_bins,
                    labels=orig_amount_bucket_labels
                    )
    
    loan_data = loan_data[loan_data['rate'] > 0]
    #loan_data = loan_data[loan_data['credit_score'] > 0 ]
    return loan_data

def prepare_survival_data(loan_data, observation_date=None):
    """
    Prepare loan data for survival analysis
    Focus: Time to Default (Approach 1)
    """
    
    # Set observation date if not provided
    if observation_date is None:
        observation_date = pd.Timestamp.now()
    else:
        observation_date = pd.to_datetime(observation_date, format='%m-%d-%Y')
    
    # Create a copy for survival analysis
    survival_data = loan_data.copy()
    
    # Define events: 1 = Default, 0 = Censored (Open or Closed)
    survival_data['event'] = (survival_data['status'] == 'DEFAULT').astype(int)
    
    # Calculate duration in months since origination
    survival_data['duration_months'] = (
        (observation_date - survival_data['open_date']).dt.days / 30.44
    ).round(2)
    
    # Basic validation
    print(f"  Observation Date: {observation_date.strftime('%Y-%m-%d')}")
    print(f"  Total Loans: {len(survival_data):,}")
    print(f"  Default Events: {survival_data['event'].sum():,} ({survival_data['event'].mean()*100:.1f}%)")
    print(f"  Censored Observations: {(1-survival_data['event']).sum():,} ({(1-survival_data['event']).mean()*100:.1f}%)")
    print(f"  Average Duration: {survival_data['duration_months'].mean():.1f} months")
    print(f"  Duration Range: {survival_data['duration_months'].min():.1f} to {survival_data['duration_months'].max():.1f} months")
    
    # Remove any negative durations (data quality issue)
    survival_data = survival_data[survival_data['duration_months'] >= 0]
    
    return survival_data

# Load Data
loan_data = structure_loan_data('loan_data.csv')
# start of Fed rate increase simulation
mid_date = datetime(2022, 4, 1)
loan_data['rate_status'] = loan_data['open_date'].apply(lambda x: 'Pre-Fed Rate Increase' if x < mid_date else 'Post-Fed Rate Increase')
survival_data = prepare_survival_data(loan_data, '01-31-2025')
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
