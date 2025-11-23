#import packages
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
from lifelines import KaplanMeierFitter


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

            risk_segment_col.append(segment)
            default_rate_col.append(round(default_rate, 1).astype(str) + '%')
            median_time_to_default_col.append(round(segment_data['duration_months'].median().astype(float), 1))
            default_size_col.append(defaults)
            sample_size_col.append(total_loans)
                        
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
            default_rate_col.append(round(default_rate, 1).astype(str) + '%')
            median_time_to_default_col.append(round(segment_data['duration_months'].median().astype(float), 1))
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
            survival_12mo_col.append(round(survival_prob_12*100, 1).astype(str) + '%')

            survival_prob_24 = kmf.survival_function_at_times(24).values[0]
            survival_24mo_col.append(round(survival_prob_24*100, 1).astype(str) + '%')

            survival_prob_36 = kmf.survival_function_at_times(36).values[0]
            survival_36mo_col.append(round(survival_prob_36*100, 1).astype(str) + '%')

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
                default_rate_col.append(round(default_rate, 1).astype(str) + '%')
                median_time_to_default_col.append(round(segment_data['duration_months'].median(), 1).astype(str))
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
                survival_12mo_col.append(round(survival_prob_12*100, 1).astype(str) + '%')

                survival_prob_24 = kmf.survival_function_at_times(24).values[0]
                survival_24mo_col.append(round(survival_prob_24*100, 1).astype(str) + '%')

                survival_prob_36 = kmf.survival_function_at_times(36).values[0]
                survival_36mo_col.append(round(survival_prob_36*100, 1).astype(str) + '%')
                
                        
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
    #survival_rate_summary.index = survival_rate_summary['Risk Segment']
    #survival_rate_summary.drop(columns=['Risk Segment'], inplace=True)

    styled_survival_rate_summary = survival_rate_summary.style\
    .set_properties(**{'color': 'black'}, **{'font-size': '14px'})

    return fig, styled_survival_rate_summary