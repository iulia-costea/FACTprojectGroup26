import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIG
# ============================================================
sns.set(style="whitegrid")

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE_BASELINE = {
    "No LLM": "#1f77b4",
    "GPT-3.5": "#ff7f0e",
}

PALETTE_ORIGINAL_CHAIN = {
    "Original": "#1f77b4",
    "Chain": "#ff7f0e",
}

PALETTE_PIPELINE = {
    "GPT-3.5 + Chain Prompting": "#1f77b4",
    "GPT-4o + Original Prompt on GPT-3.5 + Chain Prompting": "#ff7f0e",
}

# ============================================================
# FILE PATHS (UNCHANGED)
# ============================================================
files_original = {
    "No LLM": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_no_LLM.csv",
    "gpt3.5": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt3.5_first_50.csv",
    "gpt4o_mini": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt4o_mini_first_50.csv",
    "gpt4o": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\Reproducibility_Scores_UX\Qualified\first_50_scores\ScoresGoogle_UX_gpt4o_first_50.csv",
}

files_chain = {
    "gpt3.5": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt3.5_Original_File_Original_CV_first_50_optimized_2026-01-21_17-22.csv",
    "gpt4o_mini": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt4o_mini_first_50.csv",
    "gpt4o": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt4o_Original_File_Original_CV_first_260_optimized_2026-01-21_19-55.csv",
}

pipeline = {
    "GPT-3.5 + Chain": files_chain["gpt3.5"],
    "GPT-3.5 + Chain → GPT-4o": r"C:\Users\sarad\FACT\FACTprojectGroup26\Table1_Experimental_Modified_Resumes\output\input_cvs_Table1_Experimental_Modified_Resumes\Scores_chain_prompting\ScoresGoogle_UX_gpt4o_chain_on_gpt3.5_antihallucination_Original_File_file_2026-01-21_17-43.csv",
}

# ============================================================
# HELPERS
# ============================================================
def load_scores(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    score_col = [c for c in df.columns if "Score" in c][0]
    return df[score_col]


def print_violin_stats(df, group_cols, title):
    stats = (
        df.groupby(group_cols)["Resume Score"]
        .agg(
            N="count",
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Max="max",
        )
        .round(3)
        .reset_index()
    )

    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    print(stats.to_string(index=False))

    return stats


def add_medians_per_prompt(ax, df, y_col, x_col, hue_col):
    medians = df.groupby([y_col, hue_col])[x_col].median()
    categories = df[y_col].unique()
    hues = df[hue_col].unique()

    for i, cat in enumerate(categories):
        for j, hue in enumerate(hues):
            if (cat, hue) in medians:
                offset = -0.2 if j == 0 else 0.2
                ax.plot(
                    [medians[(cat, hue)], medians[(cat, hue)]],
                    [i + offset, i + offset],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---------------- FIGURE 1: BASELINE ----------------
    baseline_rows = []
    for system, label in [("No LLM", "No LLM"), ("gpt3.5", "GPT-3.5")]:
        for s in load_scores(files_original[system]):
            baseline_rows.append(
                {"Group": "Baseline", "LLM": label, "Resume Score": s}
            )

    df_baseline = pd.DataFrame(baseline_rows)

    print_violin_stats(
        df_baseline,
        ["LLM"],
        "Baseline Violin Statistics: No LLM vs GPT-3.5",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(
        data=df_baseline,
        x="Resume Score",
        hue="LLM",
        inner="quart",
        density_norm="width",
        linewidth=1,
        split=True,
        palette=PALETTE_BASELINE,
        orient="h",
        cut=0,
        ax=ax,
    )
    add_medians_per_prompt(ax, df_baseline, "Group", "Resume Score", "LLM")
    ax.set_title("Baseline Comparison")
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/violin_baseline.png", dpi=300)
    plt.close()

    # ---------------- FIGURE 2: ORIGINAL vs CHAIN ----------------
    rows = []
    for model, f_orig, f_chain in [
        ("GPT-3.5", files_original["gpt3.5"], files_chain["gpt3.5"]),
        ("GPT-4o-mini", files_original["gpt4o_mini"], files_chain["gpt4o_mini"]),
        ("GPT-4o", files_original["gpt4o"], files_chain["gpt4o"]),
    ]:
        for prompt, path in [("Original", f_orig), ("Chain", f_chain)]:
            for s in load_scores(path):
                rows.append(
                    {"LLM": model, "Prompt": prompt, "Resume Score": s}
                )

    df_plot = pd.DataFrame(rows)

    print_violin_stats(
        df_plot,
        ["LLM", "Prompt"],
        "Original vs Chain Violin Statistics",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df_plot,
        x="Resume Score",
        y="LLM",
        inner="quart",
        density_norm="width",
        linewidth=1,
        hue="Prompt",
        split=True,
        palette=PALETTE_ORIGINAL_CHAIN,
        orient="h",
        cut=0,
        ax=ax,
    )
    add_medians_per_prompt(ax, df_plot, "LLM", "Resume Score", "Prompt")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/violin_original_vs_chain.png", dpi=300)
    plt.close()

    # ---------------- FIGURE 3: PIPELINE ----------------
    rows = []
    for name, path, label in [
        ("Pipeline", pipeline["GPT-3.5 + Chain"], "GPT-3.5 + Chain Prompting"),
        (
            "Pipeline",
            pipeline["GPT-3.5 + Chain → GPT-4o"],
            "GPT-4o + Original Prompt on GPT-3.5 + Chain Prompting",
        ),
    ]:
        for s in load_scores(path):
            rows.append(
                {"Pipeline": name, "Prompt": label, "Resume Score": s}
            )

    df_pipeline = pd.DataFrame(rows)

    print_violin_stats(
        df_pipeline,
        ["Prompt"],
        "Pipeline Violin Statistics",
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.violinplot(
        data=df_pipeline,
        x="Resume Score",
        hue="Prompt",
        inner="quart",
        density_norm="width",
        linewidth=1,
        split=True,
        palette=PALETTE_PIPELINE,
        orient="h",
        cut=0,
        ax=ax,
    )
    add_medians_per_prompt(ax, df_pipeline, "Pipeline", "Resume Score", "Prompt")
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/violin_pipeline_comparison.png", dpi=300)
    plt.close()

    print("\nAll violin plots saved in ./figures/")
