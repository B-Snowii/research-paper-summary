import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import requests  # Add this import for API calls

# Path to your Excel file
excel_path = r""

# Read the Excel file
df = pd.read_excel(excel_path, header=None)

# Extract rows
title = df.iloc[0, 0]      # First row, first column
abstract = df.iloc[1, 0]   # Second row, first column
keywords = df.iloc[2, 0]   # Third row, first column
summary = df.iloc[-1, 0]   # Last row, first column

# You can now use these variables as your dataset
dataset = {
    "title": title,
    "abstract": abstract,
    "keywords": keywords,
    "summary": summary
}

# Path to your sample file (same directory as this script)
sample_path = os.path.join(os.path.dirname(__file__), "sample.xlsx")

# Read the Excel file, no header, so all rows are data
df_sample = pd.read_excel(sample_path, header=None)

# Extract columns: Title (B), Abstract (I), Keywords (J), skipping the first row
title_ideal = df_sample.iloc[1:, 1]      # Column B (index 1)
abstracts_ideal = df_sample.iloc[1:, 8]   # Column I (index 8)
keywords_ideal = df_sample.iloc[1:, 9]    # Column J (index 9)

# Add DeepSeek API call function
def deepseek_generate_summary(prompt, api_url, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    # The DeepSeek API returns the result in choices[0]['message']['content']
    return response.json()["choices"][0]["message"]["content"]

# Set your DeepSeek API endpoint and key
api_url = "https://api.deepseek.com/v1/chat/completions"  # Updated to official DeepSeek endpoint
api_key = ""           # Replace with your API key externally (e.g., environment variable)

# Prepare few-shot examples from your training data
few_shot_examples = ""
for i in range(2):  # Use as many as you want for shots
    ex_title = str(df.iloc[i, 0])
    ex_abstract = str(df.iloc[i, 1])
    ex_keywords = str(df.iloc[i, 2])
    ex_summary = str(df.iloc[i, 3])
    few_shot_examples += (
        f"Title: {ex_title}\n"
        f"Abstract: {ex_abstract}\n"
        f"Keywords: {ex_keywords}\n"
        f"Summary: {ex_summary}\n\n"
    )

# Instruction for the model, matching your ChatGPT prompt style
instruction = (
    "Below are several examples of ideal summaries from previous data. "
    "Please carefully study these samples and generate a new summary for the given paper, "
    "emulating the style, tone, and quality of the former examples. "
    "Use only third-person language (do not use 'we', 'our', or first-person pronouns). "
    "Refer to the paper by its title, abstract, and keywords. "
    "Your summary must be clear, formal, and neutral, and concise. A summary of approximately 25 words is preferred."
)

# Generate summaries for each row in sample.xlsx and write to column L (index 11)
summaries = []
for idx in range(1, len(df_sample)):
    title = str(df_sample.iloc[idx, 1])      # Column B (index 1)
    abstract = str(df_sample.iloc[idx, 8])   # Column I (index 8)
    keywords = str(df_sample.iloc[idx, 9])   # Column J (index 9)
    # Optionally truncate abstract for prompt length
    max_abstract_length = 600
    if len(abstract) > max_abstract_length:
        abstract = abstract[:max_abstract_length] + "..."
    prompt = (
        instruction +
        few_shot_examples +
        f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}\nSummary:"
    )
    summary = deepseek_generate_summary(prompt, api_url, api_key)
    # Clean up output
    if "Summary:" in summary:
        summary = summary.split("Summary:")[-1].strip()
    summaries.append(summary)
    print(f"Processed row {idx}: {summary}")

# Ensure df_sample has at least 12 columns (index 0-11)
if df_sample.shape[1] < 12:
    for _ in range(12 - df_sample.shape[1]):
        df_sample[df_sample.shape[1]] = ""

# Write summaries to column L (index 11), starting from the second row
df_sample.iloc[1:, 11] = summaries

# Save the updated DataFrame back to Excel
df_sample.to_excel(os.path.join(os.path.dirname(__file__), "sample_with_summaries.xlsx"), header=None, index=False)
