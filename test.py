import pandas as pd
df_31 = pd.read_csv("data/llama31_responses.csv")
df_33 = pd.read_csv("data/llama33_responses.csv")

print(df_31.describe())