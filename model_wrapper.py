from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SmallLLM:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            print("⚠️ Failed to load LLM:", e)
            self.model = None
            self.tokenizer = None

    def generate(self, prompt: str, max_length=128):
        if not self.model or not self.tokenizer:
            return "LLM unavailable"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text
