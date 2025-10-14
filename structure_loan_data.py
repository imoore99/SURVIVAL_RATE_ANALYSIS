
import pandas as pd


def structure_loan_data(loan_data_csv):
   
    #Structure the raw loan data into a DataFrame with specific columns.

    def bucket_loan_data(loan_data, col, bucket_col, bins, labels):
        
        #Bucket loan data into categories based on rate.
        
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