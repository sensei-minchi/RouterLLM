from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

model_name = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(model_name)

def encode_response(resp):
    embedding = model.encode(
        resp,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding

def create_np(csv_location, np_location):
    df = pd.read_csv(csv_location)
    minn = 0
    embd = []
    if os.path.exists(np_location):
        arr = np.load(np_location)
        embd = arr.tolist()
        minn = len(arr)


    for index, rows in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Generating Embeddings"
            ):
        if index < minn:
            continue
        encoded_resp = encode_response(rows["prompt"])
        embd.append(encoded_resp)

    embd = np.asarray(embd, dtype=np.float32)
    np.save(np_location, embd)
    print(f"{len(embd)} responses converted to embeddings")


llm31_csv = "C://programming/Projects/Router LLM/data/llama31_responses.csv"
llm33_csv = "C://programming/Projects/Router LLM/data/llama33_responses.csv"
prompt = "C:\\programming\\Projects\\Router LLM\\data\\router_dataset.csv"
prompts_np = "prompts.npy"

llm31_np = "llm31.npy"
llm33_np = "llm33.npy"

# create_np(llm31_csv, llm31_np)
# create_np(llm33_csv, llm33_np)
create_np(prompt, prompts_np)