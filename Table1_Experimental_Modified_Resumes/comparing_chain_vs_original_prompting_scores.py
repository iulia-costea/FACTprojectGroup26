import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- File paths ---
# Original scores
files_original = {
    "gpt3.5": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt3.5_first_50.csv",
    "gpt4o_mini": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt4o_mini_first_50.csv",
    "gpt4o": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt4o_first_50.csv"
}

# Chain prompting scores
files_chain = {
    "gpt3.5": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt3.5_Original_File_Original_CV_first_50_optimized_2026-01-21_17-22.csv",
    "gpt4o_mini": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt4o_mini_first_50.csv",
    "gpt4o": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt4o_Original_File_Original_CV_first_260_optimized_2026-01-21_19-55.csv"
}

# --- Function to load scores safely ---
def load_score(file_path):
    df = pd.read_csv(file_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Find the column that contains 'Score'
    score_col = [col for col in df.columns if "Score" in col][0]
    return df[score_col]

# --- Load all scores ---
scores_original = {k: load_score(v) for k, v in files_original.items()}
scores_chain = {k: load_score(v) for k, v in files_chain.items()}

# --- Compute basic statistics ---
print("=== Original Scores ===")
for model, scores in scores_original.items():
    print(f"{model}: mean={scores.mean():.2f}, std={scores.std():.2f}, min={scores.min():.2f}, max={scores.max():.2f}")

print("\n=== Chain Prompting Scores ===")
for model, scores in scores_chain.items():
    print(f"{model}: mean={scores.mean():.2f}, std={scores.std():.2f}, min={scores.min():.2f}, max={scores.max():.2f}")

# --- Prepare data for plotting ---
plot_data = []
for model in files_original.keys():
    for i, val in enumerate(scores_original[model]):
        plot_data.append({'Model': model, 'Version': 'Original', 'Resume': i+1, 'Score': val})
    for i, val in enumerate(scores_chain[model]):
        plot_data.append({'Model': model, 'Version': 'Chain Prompting', 'Resume': i+1, 'Score': val})

df_plot = pd.DataFrame(plot_data)

# --- Plot scores comparison ---
sns.set(style="whitegrid")
plt.figure(figsize=(12,6))
sns.lineplot(data=df_plot, x='Resume', y='Score', hue='Version', style='Model', markers=True, dashes=False)
plt.title("Comparison of Resume Scores: Original vs Chain Prompting")
plt.xlabel("Resume Index")
plt.ylabel("Score")
plt.legend(title="Version / Model")
plt.tight_layout()
plt.show()

# --- Optional: Compute differences between original and chain ---
print("\n=== Differences (Chain - Original) ===")
for model in scores_original.keys():
    diff = scores_chain[model].values - scores_original[model].values
    print(f"{model}: mean_diff={diff.mean():.3f}, std_diff={diff.std():.3f}, max_diff={diff.max():.3f}, min_diff={diff.min():.3f}")
