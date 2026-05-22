A smart router that will route prompt on the basis of prompt complexity. Easier prompts will be prompted to quicker LLM, and the prompts that are tough and require a smarter response will be routed to the LLM that can tackle complex problems easily. 

Execution Plan:-
1) Create a dataset from 5 different datasets, each one having a different variety of questions
2) Pass each prompt to llm, record each response (as we are using API, the latency will be affected by traffic but still we will record it)
3) Using LLM as a Judge, we will rate each response on the scale of 0-1 = quality_score
4) Assign score for each prompt :- score = a*(quality_score) - b*(model_weight) (model weight is an artificial weight assigned to each LLM to simulate cost per token and latency, we are using only FREE APIs here)
5) Encode each prompt, and then train a model with input X = [ embedding,prompt_length,num_questions,has_code,predicted_length,grade_level,has_math,num_constraints] to give [score] as the output
6) Make this model for each LLM and then use it to compute the score for a new prompt, the llm with the highest score will be the one to which the prompt will be routed

Testing Plan :-
1) Create a test set, route them to each model and compute score
2) Use router, compute score and compare.


LLMS Used:
1) Gemma :- *gemma-3-27b-it*
2) Llama 3.1 :- *meta-llama/llama-3.1-8b-instruct*
3) Llama 3.3 :- *meta-llama/llama-3.3-70b-instruct*