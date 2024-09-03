import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')

tokenizer.pad_token = tokenizer.eos_token

x = input("說點啥: ")
inputs = tokenizer.encode(x, return_tensors='pt', padding=True)
attention_mask = torch.ones(inputs.shape, dtype=torch.long)

outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_length=100,
    do_sample=True,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
