#importing the required packages and libraries
from models.judge import GeminiJudge
from tqdm import tqdm
import pandas as pd
import numpy as np

#locations
prompts = "data/router_dataset.csv"
llama31_responses = "data/llama31_responses.csv"
llama33_responses = "data/llama33_responses.csv"

#Judge
judge = GeminiJudge()

#Loading datasets
df = pd.read_csv(prompts)
df_llm31 = pd.read_csv(llama31_responses)
df_llm33 = pd.read_csv(llama33_responses)


#==============================Llama-3.1-8b Responses=========================

