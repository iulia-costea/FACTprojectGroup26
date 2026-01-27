import pandas as pd

# Read the score files
original_scores = pd.read_csv('tpr_calculation_files/ScoresGoogle_UX_Original_File.csv', index_col=0)
once_modified_scores = pd.read_csv('tpr_calculation_files/ScoresGoogle_UX_gpt4o.csv', index_col=0)
twice_modified_scores = pd.read_csv('tpr_calculation_files/ScoresGoogle_UX_gpt4o_on_gpt4o.csv', index_col=0)

# Read the labels file (first 260 rows only)
labels_df = pd.read_csv('Table1_Experimental_Modified_Resumes/Scores/GoogleUX_Scores.csv', index_col=0)
labels_df = labels_df.head(260)

# Get the column names (first column in each dataframe)
print("Original scores columns:", original_scores.columns.tolist())
print("Once modified columns:", once_modified_scores.columns.tolist())
print("Twice modified columns:", twice_modified_scores.columns.tolist())

# Create combined dataframe
combined_df = pd.DataFrame()
combined_df['CVGoogle_UX Score'] = original_scores.iloc[:, 0]
combined_df['Cleaned GPT-4o Conversation-Improved CVGoogle_UX Score'] = once_modified_scores.iloc[:, 0]
combined_df['Twice GPT-4o Google_UX Score'] = twice_modified_scores.iloc[:, 0]
combined_df['UX True Label'] = labels_df['UX True Label'].values
combined_df['Will Manipulate'] = labels_df['Will Manipulate'].values

# Save to file
output_file = 'tpr_calculation_files/GoogleUX_Reproducibility_Case1.csv'
combined_df.to_csv(output_file)

print(f"\nCombined data saved to {output_file}")
print(f"\nDataframe shape: {combined_df.shape}")
print(f"\nFirst few rows:")
print(combined_df.head())
print(f"\nColumn names:")
print(combined_df.columns.tolist())
