import os
import pandas as pd
import numpy as np

def generate_dummy_data():
    # 1. Define exactly 200 rows and extract the columns
    n_rows = 200
    
    columns_string = "id,member_id,loan_amnt,funded_amnt,funded_amnt_inv,term,int_rate,installment,grade,sub_grade,emp_length,home_ownership,annual_inc,verification_status,issue_d,pymnt_plan,purpose,zip_code,addr_state,dti,delinq_2yrs,earliest_cr_line,fico_range_low,fico_range_high,inq_last_6mths,mths_since_last_delinq,mths_since_last_record,open_acc,pub_rec,revol_bal,revol_util,total_acc,initial_list_status,out_prncp,out_prncp_inv,total_pymnt,total_pymnt_inv,total_rec_prncp,total_rec_int,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_d,last_pymnt_amnt,next_pymnt_d,last_credit_pull_d,last_fico_range_high,last_fico_range_low,collections_12_mths_ex_med,mths_since_last_major_derog,policy_code,application_type,annual_inc_joint,dti_joint,verification_status_joint,acc_now_delinq,tot_coll_amt,tot_cur_bal,open_acc_6m,open_act_il,open_il_12m,open_il_24m,mths_since_rcnt_il,total_bal_il,il_util,open_rv_12m,open_rv_24m,max_bal_bc,all_util,total_rev_hi_lim,inq_fi,total_cu_tl,inq_last_12m,acc_open_past_24mths,avg_cur_bal,bc_open_to_buy,bc_util,chargeoff_within_12_mths,delinq_amnt,mo_sin_old_il_acct,mo_sin_old_rev_tl_op,mo_sin_rcnt_rev_tl_op,mo_sin_rcnt_tl,mort_acc,mths_since_recent_bc,mths_since_recent_bc_dlq,mths_since_recent_inq,mths_since_recent_revol_delinq,num_accts_ever_120_pd,num_actv_bc_tl,num_actv_rev_tl,num_bc_sats,num_bc_tl,num_il_tl,num_op_rev_tl,num_rev_accts,num_rev_tl_bal_gt_0,num_sats,num_tl_120dpd_2m,num_tl_30dpd,num_tl_90g_dpd_24m,num_tl_op_past_12m,pct_tl_nvr_dlq,percent_bc_gt_75,pub_rec_bankruptcies,tax_liens,tot_hi_cred_lim,total_bal_ex_mort,total_bc_limit,total_il_high_credit_limit,revol_bal_joint,sec_app_fico_range_low,sec_app_fico_range_high,sec_app_earliest_cr_line,sec_app_inq_last_6mths,sec_app_mort_acc,sec_app_open_acc,sec_app_revol_util,sec_app_open_act_il,sec_app_num_rev_accts,sec_app_chargeoff_within_12_mths,sec_app_collections_12_mths_ex_med,sec_app_mths_since_last_major_derog,hardship_flag,hardship_type,hardship_reason,hardship_status,deferral_term,hardship_amount,hardship_start_date,hardship_end_date,payment_plan_start_date,hardship_length,hardship_dpd,hardship_loan_status,orig_projected_additional_accrued_interest,hardship_payoff_balance_amount,hardship_last_payment_amount,disbursement_method,debt_settlement_flag,debt_settlement_flag_date,settlement_status,settlement_date,settlement_amount,settlement_percentage,settlement_term,target"
    columns = columns_string.split(',')
    
    # Set seed for reproducibility
    np.random.seed(42)
    df = pd.DataFrame()

    # 2. Fill baseline feature columns with roughly scaled values
    for col in columns:
        if col == 'target':
            # Binary values with ~20% distribution of 1s
            df[col] = np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])
        elif col == 'id':
            # Sequential IDs
            df[col] = np.arange(68000000, 68000000 + n_rows)
        elif any(keyword in col for keyword in ['amnt', 'bal', 'inc', 'lim', 'prncp']):
            # Large monetary amounts
            df[col] = np.random.uniform(1000, 100000, size=n_rows).round(2)
        elif any(keyword in col for keyword in ['rate', 'percent', 'util', 'dti']):
            # Percentages and rates
            df[col] = np.random.uniform(0, 100, size=n_rows).round(2)
        elif 'fico' in col:
            # FICO scores
            df[col] = np.random.randint(600, 850, size=n_rows).astype(float)
        else:
            # Default small integers / categorical indicators
            df[col] = np.random.randint(0, 50, size=n_rows).astype(float)

    # 3. Inject 98% sparsity into specific columns to test the filter
    # 98% of 200 = 196 rows must be 0 or NaN
    num_sparse = int(n_rows * 0.98) 
    
    # Choosing three columns that realistically contain sparse data in loan datasets
    sparse_cols = ['member_id', 'hardship_amount', 'settlement_amount']
    
    for col in sparse_cols:
        # Generate exactly 196 values that are either 0.0 or NaN
        sparse_values = np.random.choice([0.0, np.nan], size=num_sparse)
        
        # Generate the remaining 4 valid, non-zero values
        real_values = np.random.uniform(500, 5000, size=n_rows - num_sparse).round(2)
        
        # Combine and shuffle so the sparsity is randomly distributed
        combined = np.concatenate([sparse_values, real_values])
        np.random.shuffle(combined)
        
        df[col] = combined

    # 4. Save to relative path
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sample_test_dataset.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Success! Generated {n_rows} rows and {len(columns)} columns.")
    print(f"Target distribution (1s): {(df['target'].sum() / n_rows) * 100:.1f}%")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_dummy_data()