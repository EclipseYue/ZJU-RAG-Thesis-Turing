import time
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...")
    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)
    print(f"Loaded in {time.time()-t0:.2f}s on {device}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": "Based on the text 'Elon Musk founded SpaceX in 2002', who founded SpaceX?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    t1 = time.time()
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=20)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated in {time.time()-t1:.2f}s: {response}")
except Exception as e:
    print(f"Error: {e}")
