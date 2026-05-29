import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

X_loc = "C:\\programming\\Projects\\Router LLM\\encoder\\prompts.npy"
y1_loc = "C:\\programming\\Projects\\Router LLM\\data\\llama31_responses.csv"
y2_loc = "C:\\programming\\Projects\\Router LLM\\data\\llama33_responses.csv"
prompts_loc = "C:\\programming\\Projects\\Router LLM\\data\\router_dataset.csv"

y1 = pd.read_csv(y1_loc)["quality_score"].to_numpy()
y2 = pd.read_csv(y2_loc)["quality_score"].to_numpy()

training_examples = min(len(y1), len(y2))

y1 = y1[:training_examples]
y2 = y2[:training_examples]
X = np.load(X_loc)[:training_examples]


