import pandas as pd

# Load original CSV
df = pd.read_csv(
    r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Original_CV_first_260.csv",
    index_col=0
)

# Keep only the first resume
df_one = df.iloc[[0]]

# Save test CSV
df_one.to_csv(
    r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Original_CV_test_1.csv"
)

print("Saved test CSV with 1 resume.")
