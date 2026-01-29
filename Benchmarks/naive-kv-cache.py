import time 
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch 

if(torch.backends.mps.is_available()):
    device ="mps"
else:
    device="cpu"

print(f"using {device} right now")

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

## generating a single token at a time 
def generate_naive_token(prompt,max_new_tokens=20):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_key_values = None # start with empty
    generated = input_ids # the stuff which is generated 
    start_time = time.perf_counter()
    prompt_len = input_ids.shape[1]

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids = generated if past_key_values is None else generated[:,-1:],
                past_key_values = past_key_values,
                use_cache=True
            )
        
        logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values ## this is the naive KV cache which grows forever

        next_token = torch.argmax(logits,dim=-1,keepdim=True)
        generated = torch.cat((generated,next_token),dim=-1)

        total_tokens = generated.shape[-1]
        num_layers=len(past_key_values)
        kv_seq_len = past_key_values.get_seq_length()

# (batch_size, num_heads, seq_len, head_dim)

        print(
            f"[Step {step:02d}]"
            f"total tokens : {total_tokens} "
            f"kv seq_len : {kv_seq_len}"
        )

        elapsed = time.perf_counter() - start_time
    return generated, elapsed, prompt_len
    

## multi turn simulation
conversation = [
    "You are a helpful assistant specialized in machine learning and systems.",
    "Explain what a key-value (KV) cache is in transformer models.",
    "Explain it again, but assume I only know basic deep learning.",
    "Give a real-world analogy that maps queries, keys, and values clearly.",
    "Now explain how KV cache works specifically during autoregressive text generation.",
    "What exactly is recomputed every token if KV cache is NOT used?",
    "What changes internally when KV cache IS enabled?",
    "Why does the time complexity change from O(n^2) to O(n) per token?",
    "Summarize everything about KV cache in 5 concise bullet points."
]


full_prompt = ""

for turn,user_input in enumerate(conversation):
    full_prompt += user_input + "\n"
    print(f"\n=== Turn {turn+1} ===")
    generated, elapsed, prompt_len = generate_naive_token(full_prompt,max_new_tokens=10)
    print(f"turn latency is : {elapsed:.2f} seconds")

    if device=="mps":
        alloc = torch.mps.current_allocated_memory()/1024**2
        print(f"current allocated memory on MPS : {alloc:.2f} MB")
new_tokens = generated[:, prompt_len:]
response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
print("\n LLM response:")
print(response)
