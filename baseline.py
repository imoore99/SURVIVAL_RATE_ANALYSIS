

import pandas as pd

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