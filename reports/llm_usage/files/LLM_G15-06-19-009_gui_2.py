# --- STEP 1: DATA UPLOAD ---
st.header("Step 1: Dataset Upload")
uploaded_file = st.file_uploader("Upload your raw dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv(PATHS["uploaded_csv"], index=False)
    st.success(f"Dataset successfully uploaded and saved to `{PATHS['uploaded_csv']}`.")
    
    with st.expander("Preview Dataset"):
        # Explicitly state the total dimensions of the loaded dataset
        total_rows, total_cols = df.shape
        st.info(f"Dataset loaded successfully. Total rows: {total_rows} | Total columns: {total_cols}")
        
        # Clarify that the table below is only a partial view
        st.caption("Showing a preview of the first 5 rows:")
        st.dataframe(df.head())