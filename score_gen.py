#importing the required packages and libraries
from models.judge import LlamaJudge
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import re

#locations
prompts = "data/router_dataset.csv"
llama31_responses = "data/llama31_responses.csv"
llama33_responses = "data/llama33_responses.csv"

#Judge
judge = LlamaJudge()

#Loading datasets
df = pd.read_csv(prompts)
df_llm31 = pd.read_csv(llama31_responses)
df_llm33 = pd.read_csv(llama33_responses)


def parse_score(score_val):
    try:
        return float(score_val)
    except (ValueError, TypeError):
        # Try extracting the first number from string
        match = re.search(r"\d+(?:\.\d+)?", str(score_val))
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not parse score: {score_val}")


def is_score_missing(val):
    if pd.isna(val):
        return True
    if val is None:
        return True
    val_str = str(val).strip().lower()
    if val_str in ("", "nan", "none", "null"):
        return True
    return False


def evaluate_with_retry(prompt, response, judge_obj, retries=5, initial_backoff=2):
    backoff = initial_backoff
    for attempt in range(retries):
        try:
            judgement = judge_obj.judge(prompt, response)
            if "score" in judgement:
                return parse_score(judgement["score"])
            elif "error" in judgement:
                raise Exception(judgement["error"])
            else:
                raise Exception("Missing 'score' key in judgement response")
        except Exception as e:
            err_msg = str(e)
            print(f"\n[Attempt {attempt + 1}/{retries}] Error judging: {err_msg}")
            
            # Check for 429 / rate limits and parse required sleep time
            if "429" in err_msg or "rate_limit" in err_msg or "Rate limit" in err_msg:
                match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_msg)
                if match:
                    wait_time = float(match.group(1)) + 2.0
                    print(f"Rate limit hit. Waiting {wait_time}s as requested by API...")
                    time.sleep(wait_time)
                    backoff = initial_backoff  # reset backoff after waiting requested time
                    continue
                else:
                    # If we can't parse the time, sleep for a default safety time of 15 seconds
                    print("Rate limit hit (could not parse wait time). Waiting 15s...")
                    time.sleep(15.0)
                    continue

            if attempt == retries - 1:
                raise e
            
            print(f"Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2


#==============================Llama-3.1-8b Responses=========================
print("Judging llama-3.1 responses")

completed_31 = set(df_llm31[~df_llm31["quality_score"].apply(is_score_missing)]["prompt_index"])

for idx, row in tqdm(
        df_llm31.iterrows(),
        total=len(df_llm31),
        desc="Judging llama 3.1 Responses"
):
    if row["prompt_index"] in completed_31:
        continue
    
    try:
        prompt = df.iloc[row["prompt_index"]]["prompt"]
        response = row["responses"]
        score = evaluate_with_retry(prompt, response, judge)
        df_llm31.at[idx, "quality_score"] = score
        df_llm31.to_csv(llama31_responses, index=False)
        completed_31.add(row["prompt_index"])
        time.sleep(5.0)  # Safe rate limit delay
    except Exception as e:
        print(f"Failed to judge llama-3.1 row {idx} after retries: {e}")
        print("Stopping execution to prevent persistent error loop.")
        break


#==============================Llama-3.3-70b Responses=========================
print("Judging llama-3.3 responses")

# Re-read df_llm33 to make sure we have any partially completed progress if we resumed
df_llm33 = pd.read_csv(llama33_responses)

completed_33 = set(df_llm33[~df_llm33["quality_score"].apply(is_score_missing)]["prompt_index"])

for idx, row in tqdm(
        df_llm33.iterrows(),
        total=len(df_llm33),
        desc="Judging llama 3.3 Responses"
):
    if row["prompt_index"] in completed_33:
        continue
    
    try:
        prompt = df.iloc[row["prompt_index"]]["prompt"]
        response = row["responses"]
        score = evaluate_with_retry(prompt, response, judge)
        df_llm33.at[idx, "quality_score"] = score
        df_llm33.to_csv(llama33_responses, index=False)
        completed_33.add(row["prompt_index"])
        time.sleep(5.0)  # Safe rate limit delay
    except Exception as e:
        print(f"Failed to judge llama-3.3 row {idx} after retries: {e}")
        print("Stopping execution to prevent persistent error loop.")
        break