import random
import pandas as pd
from datasets import load_dataset


NUM_PER_DATASET = 200
random.seed(42)

all_prompts = []


def add_prompts(dataset_name, prompts, category):
    for prompt in prompts:
        all_prompts.append({
            "prompt": prompt,
            "source": dataset_name,
            "category": category
        })


# =========================
# GSM8K
# =========================
gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split="train"
)

gsm8k_samples = random.sample(
    list(gsm8k),
    min(NUM_PER_DATASET, len(gsm8k))
)

gsm8k_prompts = [x["question"] for x in gsm8k_samples]

add_prompts("gsm8k", gsm8k_prompts, "math")


# =========================
# 2. HumanEval
# =========================
humaneval = load_dataset(
    "openai/openai_humaneval",
    split="test"
)

humaneval_samples = random.sample(
    list(humaneval),
    min(NUM_PER_DATASET, len(humaneval))
)

humaneval_prompts = [x["prompt"] for x in humaneval_samples]

add_prompts("humaneval", humaneval_prompts, "coding")


# =========================
# 3. MBPP
# =========================
mbpp = load_dataset(
    "google-research-datasets/mbpp",
    split="train"
)

mbpp_samples = random.sample(
    list(mbpp),
    min(NUM_PER_DATASET, len(mbpp))
)

mbpp_prompts = [x["text"] for x in mbpp_samples]

add_prompts("mbpp", mbpp_prompts, "coding")


# =========================
# 4. UltraChat
# =========================
ultrachat = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft"
)

ultrachat_samples = random.sample(
    list(ultrachat),
    min(NUM_PER_DATASET, len(ultrachat))
)

ultrachat_prompts = []

for item in ultrachat_samples:
    messages = item["messages"]

    if len(messages) > 0:
        ultrachat_prompts.append(messages[0]["content"])

add_prompts("ultrachat", ultrachat_prompts, "chat")


# # =========================
# # 5. GPQA
# # =========================
# gpqa = load_dataset(
#     "Idavidrein/gpqa",
#     "gpqa_main",
#     split="train"
# )

# gpqa_samples = random.sample(
#     list(gpqa),
#     min(NUM_PER_DATASET, len(gpqa))
# )

# gpqa_prompts = [x["Question"] for x in gpqa_samples]

# add_prompts("gpqa", gpqa_prompts, "reasoning")


# =========================
# 6. BBH
# =========================
bbh = load_dataset(
    "lukaemon/bbh",
    "temporal_sequences",
    split="test"
)

bbh_samples = random.sample(
    list(bbh),
    min(NUM_PER_DATASET, len(bbh))
)

bbh_prompts = [x["input"] for x in bbh_samples]

add_prompts("bbh", bbh_prompts, "reasoning")


# =========================
# Shuffle Everything
# =========================
random.shuffle(all_prompts)


# =========================
# Save CSV
# =========================
df = pd.DataFrame(all_prompts)

df.to_csv("router_dataset.csv", index=False)

print("Saved dataset to router_dataset.csv")
print("Total prompts:", len(df))