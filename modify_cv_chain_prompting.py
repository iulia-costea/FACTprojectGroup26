"""
modify_cv_chain_openrouter.py

Improves resumes using OpenRouter API with chain prompting and checkpointing.

Chain steps:
1. Keyword alignment
2. Subheadings
3. Bullet points / formatting
4. Relevance filtering / anti-hallucination

Each step is applied sequentially on the output of the previous step.
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import os


# -------------------- Chain Prompts --------------------

ANTI_HALLUCINATION = (
    "I will be punished if I lie about my qualifications, so it is imperative "
    "that the new resume does not add any facts that are not in the original resume. "
    "Also, please do not include any additional notes or explanation: I just want the text of the new resume."
)

CHAIN_PROMPTS = [
    f"""
You are an applicant rewriting your resume for an ATS.

ORIGINAL RESUME:
{{original_cv}}

JOB DESCRIPTION:
{{job_description}}

TASK:
Change the wording in the resume to incorporate keywords used in the job description.

{ANTI_HALLUCINATION}
""",

    f"""
You are an applicant rewriting your resume for an ATS.

ORIGINAL RESUME:
{{original_cv}}

TASK:
Modify the resume to use traditional subheadings such as Professional Summary, Skills, Experience, and Education.

{ANTI_HALLUCINATION}
""",

    f"""
You are an applicant rewriting your resume for an ATS.

ORIGINAL RESUME:
{{original_cv}}

TASK:
Modify the resume to use bullet points and short sentences (except for in the professional summary).

{ANTI_HALLUCINATION}
""",

    f"""
You are an applicant rewriting your resume for an ATS.

ORIGINAL RESUME:
{{original_cv}}

TASK:
Remove side jobs or non-relevant experience.

{ANTI_HALLUCINATION}
"""
]

# -------------------- Helper Functions --------------------

def format_prompt_messages(template: str, input_cv: str, job_description: str = None):
    """Format messages for OpenRouter API."""
    if job_description:
        content = template.replace("{original_cv}", input_cv).replace("{job_description}", job_description)
    else:
        content = template.replace("{original_cv}", input_cv)
    return [{"role": "user", "content": content}]

# -------------------- OpenRouter Client --------------------

class OpenRouterClient:
    """Simplified OpenRouter client for chain prompting."""
    def __init__(self, api_key: str, model: str = None):
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model if model else "anthropic/claude-3.5-sonnet"
        self.num_generated = 0

    def generate_single_cv(self, input_cv: str, prompt_template: str, job_description: str = None) -> str:
        """Generate one CV using the given prompt template."""
        messages = format_prompt_messages(prompt_template, input_cv, job_description)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        self.num_generated += 1
        print(f"Generated {self.num_generated} CV(s)")
        return response.choices[0].message.content

# -------------------- Chain Prompting --------------------

def chain_generate_cv(client: OpenRouterClient, input_cv: str, chain_templates: list, job_desc: str = None) -> str:
    """Apply chain prompting sequentially to a single CV."""
    current_cv = input_cv
    for template in chain_templates:
        current_cv = client.generate_single_cv(current_cv, template, job_desc)
    return current_cv

# -------------------- CLI Parsing --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Improve resumes with OpenRouter chain prompting")
    parser.add_argument("resumes", type=Path, nargs='+', help="Resume CSV files")
    parser.add_argument("outputdir", type=Path, help="Directory to save optimized resumes")
    parser.add_argument("--prompt-job-description", type=Path, help="Optional job description file")
    parser.add_argument("--model", type=str, help="OpenRouter model name")
    parser.add_argument("--api-key", type=Path, required=True, help="Path to OpenRouter API key YAML file")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N resumes")
    args = parser.parse_args()

    args.outputdir.mkdir(parents=True, exist_ok=True)
    return args

# -------------------- Main --------------------

if __name__ == "__main__":
    args = parse_args()

    # Load OpenRouter API key
    with open(args.api_key, 'r') as f:
        config = yaml.safe_load(f)
    user_api_key = config.get('openrouter', {}).get('api_key')
    if not user_api_key:
        raise ValueError("OpenRouter API key not found in YAML file")

    # Load job description if provided
    job_desc_str = None
    if args.prompt_job_description:
        job_desc_str = open(args.prompt_job_description, 'r').read()

    # Initialize OpenRouter client
    client = OpenRouterClient(api_key=user_api_key, model=args.model)

    # Process each resume CSV
    for resume_path in args.resumes:
        df = pd.read_csv(resume_path, index_col=0)
        df = df.head(50)  # only the first 50 resumes
        optimized_results = []

        for i, cv in enumerate(df[df.columns[-1]]):
            try:
                final_cv = chain_generate_cv(client, cv, CHAIN_PROMPTS, job_desc=job_desc_str)
                optimized_results.append(final_cv)
            except Exception as e:
                print(f"Error processing CV {i}: {e}")
                optimized_results.append("ERROR_GENERATING_CV")

            # Save checkpoint
            if (i + 1) % args.checkpoint_every == 0:
                checkpoint_file = Path(args.outputdir) / f"{resume_path.stem}_checkpoint_{i+1}.csv"
                pd.DataFrame({f"Optimized_{client.model}": optimized_results}).to_csv(checkpoint_file)
                print(f"Checkpoint saved: {checkpoint_file}")

        # Save final optimized CVs
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        output_file = Path(args.outputdir)/f"{resume_path.stem}_optimized_{timestamp_str}.csv"
        pd.DataFrame({f"Optimized_{client.model}": optimized_results}).to_csv(output_file)
        print(f"Saved final optimized resumes to {output_file}")
