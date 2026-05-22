from models.gemini import GeminiFlash
from models.llama_31 import Llama
import pandas as pd
from tqdm import tqdm
import os
import time

# load dataset
df = pd.read_csv("data/router_dataset.csv")


class GenerateResponses:
    def __init__(self, llm, location):
        self.llm = llm
        self.location = location

        if os.path.exists(location):
            existing_df = pd.read_csv(location)

            self.responses = existing_df["responses"].tolist()
            self.index = existing_df["prompt_index"].tolist()

            self.completed = set(self.index)

            print(f"Resuming from {len(self.responses)} responses")

        else:
            self.responses = []
            self.index = []
            self.completed = set()

    def save_progress(self):
        temp_df = pd.DataFrame({
            "prompt_index": self.index,
            "responses": self.responses,
            "quality_score": [None] * len(self.responses)
        })

        temp_df.to_csv(self.location, index=False)

    def generate(self):
        try:
            for idx, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Generating Responses"
            ):
                if idx in self.completed:
                    continue

                prompt = f"Generate a response to the following query: {row['prompt']}"

                response = self.llm.response(prompt)
                self.responses.append(response)
                self.index.append(idx)
                self.save_progress()
                time.sleep(1)

        except Exception as e:
            print(f"\nERROR OCCURRED: {e}")
            print("Saving progress before exiting...")

            self.save_progress()

        finally:
            self.save_progress()
            print(f"Saved {len(self.responses)} responses to {self.location}")


# ========================= Llama 3.1 ========================

print("=======Generating Responses from Llama 3.1=======")

location = "data/llama31_responses.csv"

llama31 = Llama("llama-3.1-8b-instant")
llama31_gen = GenerateResponses(llama31, location)
llama31_gen.generate()


# ========================= Llama 3.3 ========================

print("========Generating Llama 3.3 responses===========")

location = "data/llama33_responses.csv"

llama33 = Llama("llama-3.3-70b-versatile")
l33_gen = GenerateResponses(llama33, location)
l33_gen.generate()