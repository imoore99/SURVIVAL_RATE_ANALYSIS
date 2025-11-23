#import packages
import pandas as pd
import warnings
from lifelines import KaplanMeierFitter

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_combined_survival_analysis_plotly(survival_data, rate_period, score_tier, colors, baseline=True):
    #create table segments for dataframe (same as before)
    risk_segment_col = []
    default_rate_col = []
    median_time_to_default_col = []
    survival_12mo_col = []
    survival_24mo_col = []
    survival_36mo_col = []
    default_size_col = []
    sample_size_col = []

    # Create Plotly figure instead of matplotlib
    fig = go.Figure()

    def kmf_baseline(survival_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data['duration_months'], 
                    survival_data['event'],
                    label='Portfolio Baseline Survival Rate')
        return kmf

    # Add baseline if requested
    kmf = kmf_baseline(survival_data)
    if baseline == True:
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_['Portfolio Baseline Survival Rate'],
            mode='lines',
            name='Baseline Survival Rate',
            line=dict(color='black', width=3, dash='dash'),
            hovertemplate='<b>Baseline</b><br>' +
                         'Time: %{x:.1f} months<br>' +
                         'Survival: %{y:.1%}<br>' +
                         '<extra></extra>'
        ))

    # Handle different analysis scenarios
    if len(score_tier) == 0:
        # Rate period only analysis
        for i, rate in enumerate(rate_period):
            segment = f"{rate}"
            segment_data = survival_data[survival_data['rate_status'] == segment]
            
            # Get color for this segment
            color = next((c['color'] for c in colors if c['label'] == segment), '#1f77b4')

            # Calculate metrics (same logic as before)
            total_loans = len(segment_data)
            defaults = segment_data['event'].sum()
            default_rate = (defaults / total_loans) * 100

            risk_segment_col.append(segment)
            default_rate_col.append(round(default_rate, 1).astype(str) + '%')
            median_time_to_default_col.append(round(segment_data['duration_months'].median(), 1))
            default_size_col.append(defaults)
            sample_size_col.append(total_loans)

            # Fit KMF and add to Plotly
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=segment_data['duration_months'],
                event_observed=segment_data['event'],
                label=f'{segment}'
            )
            
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_[f'{segment}'],
                mode='lines',
                name=f'{segment} ({defaults} defaults | {round(default_rate, 1)}%)',
                line=dict(color=color, width=3),
                hovertemplate=f'<b>{segment}</b><br>' +
                             'Time: %{x:.1f} months<br>' +
                             'Survival: %{y:.1%}<br>' +
                             f'Defaults: {defaults}<br>' +
                             f'Default Rate: {default_rate:.1f}%<br>' +
                             '<extra></extra>'
            ))

    elif len(rate_period) == 0:
        # Score tier only analysis
        for i, score in enumerate(score_tier):
            segment = f"{score}"
            segment_data = survival_data[survival_data['score_bucket'] == segment]
            
            color = next((c['color'] for c in colors if c['label'] == segment), '#1f77b4')

            total_loans = len(segment_data)
            defaults = segment_data['event'].sum()
            default_rate = (defaults / total_loans) * 100

            risk_segment_col.append(segment)
            default_rate_col.append(round(default_rate, 1).astype(str) + '%')
            median_time_to_default_col.append(round(segment_data['duration_months'].median(), 1))
            default_size_col.append(defaults)
            sample_size_col.append(total_loans)

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=segment_data['duration_months'],
                event_observed=segment_data['event'],
                label=f'{segment}'
            )
            
            # Calculate survival probabilities for table
            survival_prob_12 = kmf.survival_function_at_times(12).values[0]
            survival_12mo_col.append(round(survival_prob_12*100, 1).astype(str) + '%')

            survival_prob_24 = kmf.survival_function_at_times(24).values[0]
            survival_24mo_col.append(round(survival_prob_24*100, 1).astype(str) + '%')

            survival_prob_36 = kmf.survival_function_at_times(36).values[0]
            survival_36mo_col.append(round(survival_prob_36*100, 1).astype(str) + '%')

            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_[f'{segment}'],
                mode='lines',
                name=f'{segment} ({defaults} defaults | {round(default_rate, 1)}%)',
                line=dict(color=color, width=3),
                hovertemplate=f'<b>{segment}</b><br>' +
                             'Time: %{x:.1f} months<br>' +
                             'Survival: %{y:.1%}<br>' +
                             f'Defaults: {defaults}<br>' +
                             f'Default Rate: {default_rate:.1f}%<br>' +
                             '<extra></extra>'
            ))

    else:
        # Full analysis - both rate and score
        for i, rate in enumerate(rate_period):
            for j, score in enumerate(score_tier):
                segment = f"{score}, {rate}"
                segment_data = survival_data[survival_data['risk_rate_segment'] == segment]
                
                color = next((c['color'] for c in colors if c['label'] == segment), '#1f77b4')

                total_loans = len(segment_data)
                defaults = segment_data['event'].sum()
                default_rate = (defaults / total_loans) * 100

                risk_segment_col.append(segment)
                default_rate_col.append(round(default_rate, 1).astype(str) + '%')
                median_time_to_default_col.append(round(segment_data['duration_months'].median(), 1))
                default_size_col.append(defaults)
                sample_size_col.append(total_loans)

                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=segment_data['duration_months'],
                    event_observed=segment_data['event'],
                    label=f'{segment}'
                )
                
                survival_prob_12 = kmf.survival_function_at_times(12).values[0]
                survival_12mo_col.append(round(survival_prob_12*100, 1).astype(str) + '%')

                survival_prob_24 = kmf.survival_function_at_times(24).values[0]
                survival_24mo_col.append(round(survival_prob_24*100, 1).astype(str) + '%')

                survival_prob_36 = kmf.survival_function_at_times(36).values[0]
                survival_36mo_col.append(round(survival_prob_36*100, 1).astype(str) + '%')

                fig.add_trace(go.Scatter(
                    x=kmf.timeline,
                    y=kmf.survival_function_[f'{segment}'],
                    mode='lines',
                    name=f'{segment} ({defaults} defaults | {round(default_rate, 1)}%)',
                    line=dict(color=color, width=3),
                    hovertemplate=f'<b>{segment}</b><br>' +
                                 'Time: %{x:.1f} months<br>' +
                                 'Survival: %{y:.1%}<br>' +
                                 f'Defaults: {defaults}<br>' +
                                 f'Default Rate: {default_rate:.1f}%<br>' +
                                 '<extra></extra>'
                ))

    # Add milestone markers
    max_duration = max(survival_data['duration_months'])
    
    # Add vertical lines for key milestones
    for milestone in [12, 24, 36]:
        if milestone <= max_duration:
            fig.add_vline(
                x=milestone, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                annotation_text=f"{milestone}M",
                annotation_position="top"
            )

    # Mobile-optimized layout
    fig.update_layout(
        title=dict(
            text='Survival Analysis by Credit Risk Segment',
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        xaxis=dict(
            title='Months Since Origination',
            titlefont=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title='Survival Probability (No Default)',
            titlefont=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            tickformat='.0%',
            range=[-0.02, 1.02]
        ),
        # Mobile-friendly legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        # Responsive sizing
        autosize=True,
        height=500,  # Fixed height for mobile
        margin=dict(l=50, r=150, t=60, b=50),
        hovermode='x unified',
        plot_bgcolor='white'
    )

    # Create the same summary dataframe as before
    survival_rate_summary = pd.DataFrame({
        'Risk Segment': risk_segment_col,
        'Default Rate (%)': default_rate_col,
        'Median Time to Default (months)': median_time_to_default_col,
        '12 Month Survival Rate (%)': survival_12mo_col,
        '24 Month Survival Rate (%)': survival_24mo_col,
        '36 Month Survival Rate (%)': survival_36mo_col,
        'Number of Defaults': default_size_col
    })

    styled_survival_rate_summary = survival_rate_summary.style\
        .set_properties(**{'color': 'black'}, **{'font-size': '14px'})

    return fig, styled_survival_rate_summary