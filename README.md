> ⚠️ **Repository Note**
>
> This repository is a reproduction and extension of the original codebase for  
> *Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations*  
> by Cohen et al.  
>
> In addition to reproducing the original experiments, this version includes
> implementation fixes, clarifications, and new analyses that are **not present in the original repository**.
> All original credit belongs to the paper's authors.
> Please refer to sections "Known Issues and Fixes" and "Further Work" for our contributions to the repository


## Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations

This repository contains code and experiments for the ICML paper [Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations](https://www.arxiv.org/abs/2502.13221).

## Abstract
In an era of increasingly capable foundation models, job seekers are turning to generative AI tools to
enhance their application materials. However, unequal access to and knowledge about generative AI tools
can harm both employers and candidates by reducing the accuracy of hiring decisions and giving some
candidates an unfair advantage. To address these challenges, we introduce a new variant of the strategic
classification framework tailored to manipulations performed using large language models, accommodating
varying levels of manipulations and stochastic outcomes. We propose a “two-ticket” scheme, where
the hiring algorithm applies an additional manipulation to each submitted resume and considers this
manipulated version together with the original submitted resume. We establish theoretical guarantees
for this scheme, showing improvements for both the fairness and accuracy of hiring decisions when the
true positive rate is maximized subject to a no false positives constraint. We further generalize this
approach to an n-ticket scheme and prove that hiring outcomes converge to a fixed, group-independent
decision, eliminating disparities arising from differential LLM access. Finally, we empirically validate our
framework and the performance of our two-ticket scheme on real resumes using an open-source resume
screening tool.

## Overview

This repository provides implementations for experiments in the paper to verify the theoretical improvements of our "two-ticket" scheme. We use part of the [Djinni Recruitment Dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/blob/main/README.md) dataset for sample resumes, and their matched occupations (i.e. Product Manager, UI/UX designer, etc.). We also then draw from various online job postings for the relevant job descriptions we score our resumes against. 

## Data & LLM Tools

### Djinni Recruitment Dataset

The Djinni dataset file which we used to generate results for the effectiveness of our two ticket system in Table 1 ('Table1_Experimental_Modified_Resumes/Original_CV.csv) can be downloaded from:
- [Stereotypes in Recruitment Dataset](https://github.com/Stereotypes-in-LLMs/recruitment-dataset) - Downloaded All Data, and Filtered for first 260 Product Manager and first 260 UI/UX designer resumes.

The Djinni dataset file which we used to generate results for the effectiveness of different LLM tools and their performance against different job descriptions in Figure 1 (data under 'Figure1_100Samples/Resumes/original.csv') is a subset of our above data: namely we filtered for the first 50 out of 260 Product Manager and first 50 260 UI/UX designer resumes.

### Job Descriptions Data
The two job descriptions used to generate results for resume scores in Table 1 can be found here:
- [DoorDash PM Job Description](sample_input_data/example_job_descriptions/PM_job_descriptions/doordash_pm.txt)  - the description was drawn directly from [here](https://careersatdoordash.com/jobs/product-manager-multiple-levels/5523275/).
- [Google UX Designer Job Description](sample_input_data/example_job_descriptions/UX_job_descriptions/google_ux.txt) - the job description has since been taken down from the Google Portal but was downloaded from the 2024 recruitment cycle.

The remaining job descriptions used to generate results for resume scores in Figure 1 can be found in this [folder](sample_input_data/example_job_descriptions).

### LLM Tools

We used the following LLMs to perform manipulations on our input resumes:
- GPT-3.5-Turbo-0125
- GPT-4o-mini
- GPT-4o-2024-08-06 
- Claude-3.5-Sonnet
- DeepSeek-67B
- DeepSeek-V3 DeepSeek 
- Mixtral-8x7b-Instruct
- Llama3.3-70B-Instruct-Turbo

The final paper contains details about the sequence of LLM manipulations. 

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/heyyjudes/llm-hiring-ecosystem.git
   cd llm-hiring-ecosystem
   ```

2. **Install required libraries:**
   The main dependencies are listed in `env.yml`. Install them with:
   ```
   conda env create -f env.yml
   ```

## Manipulating & Modifying Resumes

modify_cv.py is a Python script that first improves/modifies resumes/CVs using various LLM APIs given a set of inputted resumes and custom prompts. score_cv.py then scores inputted resumes against inputted job descriptions.

#### Inputs and Outputs for modify_cv.py

Running modify_cv takes in the following inputs and outputs the modified CVs in a .csv file. Optional inputs also have default values in the code:
 
##### Input Parameters

1. **Input CVs**  
   - **Type**: Filepath(s)  
   - **Required**

2. **Output Directory**  
   - **Type**: Filepath  
   - **Required**

3. **Prompt Template**  
   - **Type**: Filepath (`.json` or `.txt`)  
   - **Required**  
   - **Details**: Prompt with a placeholder for `{original_cv}` (optional placeholder for `{job_description}`). We recommend using '.txt' files if you only need to interface with the LLM-API as a "user". Otherwise, check the example-json prompt template to interface with the LLM-API as an assistant (in addition to user).

4. **Job Description for Prompt**  
   - **Type**: `.txt`  
   - **Optional**  
   - **Details**: User-inputted prompt for the last prompt type.

5. **LLM Provider**  
   - **Type**: String  
   - **Options**: `OpenAI`, `Together`, `Anthropic`  
   - **Required**

6. **API-Key**  
   - **Type**: Filepath  
   - **Required**  
   - **Details**: Path to the `api_keys.yaml` file.

7. **Model**  
   - **Type**: String  
   - **Optional**  
   - **Details**: Name of the model to use (besides default).

It outputs a csv, timestamped, with one column corresponding to the modified resume/CV text. 

To test modify_cv.py with our example files and anti-hallucination-prompt, per described in our manuscript, run in the root directory of this folder:

```bash
python modify_cv.py sample_input_data/example_input_cvs/three_example_cvs.csv sample_input_data/example_output_data --prompt-template sample_input_data/example_prompts/anti_hallucination_llm_prompt.txt --prompt-job-description sample_input_data/example_job_descriptions/PM_job_descriptions/doordash_pm.txt --provider openai --api-key api_keys.yaml
```

**Note:** Use `python` instead of `python3` to ensure you're using the conda environment. Replace `api_keys.yaml` with your actual API keys file name.

#### Inputs and Outputs for `score_cv`

The `score_cv` function takes the following inputs and outputs the scores of the inputted CVs in `.csv` format:

##### Input Parameters

1. **Input CVs**  
   - **Type**: Filepath(s)  
   - **Required**

2. **Output Directory**  
   - **Type**: Filepath  
   - **Required**

3. **Job Description**  
   - **Type**: String  
   - **Optional**  
   - **Default**: `"DoorDash PM"` [Job Description](sample_input_data/example_job_descriptions/PM_job_descriptions/doordash_pm.txt).

4. **Job Name**  
   - **Type**: String  
   - **Optional**  
   - **Default**: `"DoorDash PM"` (see above for description).

It is natural to run score_cv.py on the output resumes of modify_cv.py (and input resumes too). To test score_cv.py with our example files, run in the root directory of this folder:

```bash
python score_cv.py --resumes sample_input_data/example_input_cvs/three_example_cvs.csv --output-dir sample_input_data/example_output_data --job-description sample_input_data/example_job_descriptions/PM_job_descriptions/doordash_pm.txt --job-name DoorDash
```

## Experiments

After generating and scoring our resumes on the relevant job descriptions, we also analyzed the results with statistical inference techniques and plotted resulting scores to create Figures 1 and Tables 1 of our final manuscript. 

- **Evaluating the Two-Ticket System**: [significance_tests.ipynb](Table1_Experimental_Modified_Resumes/validation_tests/signifiance_test.ipynb) Implements our Two-Ticket Algorithm and demonstrates how it compares against traditional Threshold Classifiers (Section 7 of our paper).
- **Comparing Different LLMs**: [final_figures.ipynb](final_figures.ipynb) Code to create final figures comparing the effecitiveness of different LLMs against different job descriptions (Section 3 of our paper and Appendix Section B). 

## Citation

If you use this code or find it helpful, please cite our paper:
```
@article{cohen2025ticketsbetteronefair,
  title={Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations},
  author={Cohen, Lee and Hsieh, Jack and Hong, Connie, and Shen, Judy},
  journal={arXiv preprint arXiv:2502.13221},
  year={2025}
}
```

## Known Issues and Fixes

Issues found during code reproduction that prevented out-of-the-box execution:

1. **`score_cv.py`** - Expected `'id'` column, but CSV uses unnamed first column. Fixed by changing `index_col='id'` to `index_col=0` (line 146).

2. **`modify_cv.py`** - Expected `services -> provider -> api_key` YAML structure. Fixed by adding support for both formats (lines 546-549).

3. **Missing embeddings** - `final_figures.ipynb` requires `.npy` files in `Figure1_100Samples/Embeddings/` but no generation script exists.

4. **`consistency_checks.ipynb`** - Incorrect column names caused KeyErrors. Fixed by updating to match actual CSV structure.

5. **Incorrect README commands** - Fixed example commands to use proper argument syntax and correct file names.

6. **Python interpreter** - Changed from `python3` to `python` to ensure conda environment is used.

7. **`env.yml`** - Original environment file had dependency conflicts and incompatible package versions that prevented successful conda environment creation. Updated with compatible package versions and resolved dependency issues to enable proper installation.

## Further Work

This reproduction includes additional analyses and visualizations beyond the original repository:

### Additional Visualizations

**Claim 2 Analysis: Resume Score Distributions Under LLM Manipulations**

1. **[claim_2_doordash_pm.ipynb](claim_2_doordash_pm.ipynb)** - Generates violin plots comparing resume scores under different LLM manipulations for the DoorDash Product Manager role. Visualizes score distributions for 7 manipulation models (Claude-Sonnet-4, DeepSeek-Chat, GPT-3.5-Turbo, GPT-4o, GPT-4o-mini, Llama-3.3-70B, Mixtral-8x7B) vs no manipulation baseline, separated by qualified/unqualified status.

2. **[claim_2_google_ux.ipynb](claim_2_google_ux.ipynb)** - Same analysis as above but for the Google UX Designer role. Compares scores across the same 7 LLM manipulation models to demonstrate effects of strategic resume modifications.

**Claim 3 Analysis: Binary Classification Performance**

3. **[claim_3_google_ux.ipynb](claim_3_google_ux.ipynb)** - Reproduces Table 1 results for Google UX Designer role with full dataset (260 qualified + 260 unqualified resumes). Tests three experimental conditions comparing Traditional (1-ticket) vs Two-Ticket hiring schemes across 500 iterations. Demonstrates TPR improvements and disparity reductions from the two-ticket approach.

4. **[replicability_notebook_tpr_calculation.ipynb](replicability_notebook_tpr_calculation.ipynb)** - Extension analysis for DoorDash PM role using smaller dataset (50 qualified + 50 unqualified resumes). Performs same binary classification experiments as Claim 3 notebook but with relaxed FPR constraint (5%) due to limited sample size. Shows the two-ticket scheme's effectiveness persists across different dataset sizes.

### Prompt Engineering Comparisons

5. **[replicability_prompt_comparison.py](replicability_prompt_comparison.py)** - Compares resume scores under different prompting strategies:
   - **Baseline comparison**: No LLM manipulation vs GPT-3.5 modification
   - **Original vs Chain prompting**: Tests effectiveness of chain-of-thought prompting across GPT-3.5, GPT-4o-mini, and GPT-4o models
   - **Pipeline analysis**: Evaluates two-stage manipulation (GPT-3.5 + Chain → GPT-4o anti-hallucination)

   Generates three violin plots saved to `figures/` directory showing statistical distributions and quartiles for each condition.

6. **[claim_extension_replicability.py](claim_extension_replicability.py)** - Extended version of the replicability analysis with the same three-figure pipeline as above, using relative paths for portability.

### Path Standardization

All notebooks and scripts have been updated to use relative paths instead of absolute paths, ensuring reproducibility across different systems and directory structures. Key files updated:
- `comparing_chain_vs_original_prompting_scores.py`
- `final_replicability_graphs.py`
- `claim_extension_replicability.py`

