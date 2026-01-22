import pandas as pd

# There were more resume scores for the gpt4o-mini with chain prompting. This creates a csv file with only the first 50
# Read the original CSV (use raw string with r"" to handle backslashes)
df = pd.read_csv(r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\ScoresGoogle_UX_gpt35turbo_Original_File_file_2026-01-11_22-48.csv")

# Keep only the first 50 rows
df_first_50 = df.head(50)

# Save to a new CSV in the same folder
df_first_50.to_csv(r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt3.5_first_50.csv", index=False)