import torch
import time
cached_compiled_fns = None

@torch.no_grad()
def prefill(model, tokens):
    # Forward pass through the model
    input_pos = torch.arange(0, tokens.size(0), device="cuda", dtype=torch.int64)
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    token_id = torch.argmax(logits[0, -1])
    return token_id
    
@torch.no_grad()
def generate(model, tokens, input_pos):
    # Forward pass through the model
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    token_id = torch.argmax(logits[0, -1])
    return token_id

def compiled_fns():
    #https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.assert_indirect_indexing = False
    #prefill_compiled = torch.compile(prefill, fullgraph=True, mode="reduce-overhead")
    generate_compiled = torch.compile(generate, fullgraph=True, mode="reduce-overhead")
    return prefill, generate_compiled


def generate_text(model, prompt, compile, tokenizer, max_tokens_to_gen, terminators=None):
    if compile:
        prefill_fn, generate_fn = cached_compiled_fns or compiled_fns()
    else:
        prefill_fn, generate_fn = prefill, generate
    #print(formatted_prompt)
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze().cuda()

    # Print the initial prompt details
    start = time.time()
    tokens = prompt_tokens
    output_tokens = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        for step in range(max_tokens_to_gen):
            if step == 0: # prefill
                next_token = prefill_fn(model, tokens)
                input_pos = torch.tensor([tokens.size(0)], device="cuda", dtype=torch.int64)
                tokens = next_token.clone()
            else:
                with torch.nn.attention.sdpa_kernel(
                            torch.nn.attention.SDPBackend.MATH):
                    next_token = generate_fn(model, tokens, input_pos)
                    input_pos.add_(1)
                    tokens.copy_(next_token)
            output_tokens.append(next_token.clone())
    end = time.time()
    time_taken = end - start
    # Decode and display the final generated output
    if terminators is not None:
        truncated_output_tokens = []
        for token in output_tokens:
            if token in terminators:
                break
            truncated_output_tokens.append(token)
    else:
        truncated_output_tokens = output_tokens
    generated_text = tokenizer.decode(truncated_output_tokens)
    tokens_per_second = len(output_tokens) / time_taken
    return generated_text, tokens_per_second
