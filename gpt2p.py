from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class GPT2PNode:
    def __init__(self):
        self.models = {}
        self.tokenizer = None
        self.rephrasers = ["microsoft/Promptist"]
        pass
    @classmethod
    def INPUT_TYPES(cls):
        
       # Defines input types for the GPT-2 node
        
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),

                "model_repo": ("SELECT", {
                    "options": [
                        "Gustavosta/MagicPrompt-Stable-Diffusion",
                        "daspartho/prompt-extend",
                        "succinctly/text2image-prompt-generator", 
                        "microsoft/Promptist",
                        "AUTOMATIC/promptgen-lexart",
                        "AUTOMATIC/promptgen-majinai-safe",
                        "AUTOMATIC/promptgen-majinai-unsafe",
                        "Gustavosta/MagicPrompt-Dalle",
                        "kmewhort/stable-diffusion-prompt-bolster",
                        "Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator",
                        "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator",
                        "crumb/bloom-560m-RLHF-SD2-prompter-aesthetic",
                        "Meli/GPT2-Prompt",
                        "DrishtiSharma/StableDiffusion-Prompt-Generator-GPT-Neo-125M"
                    ]
                }),

                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("CONDITIONING","STRING",)
    FUNCTION = "generate"
    CATEGORY = "conditioning"

    def generate(self, clip, text, model_repo, temperature):
        if self.tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        num_return_sequences = 1
        max_length = 100 # max: 512

        # Load the model based on the model_repo
        if model_repo not in self.models:
            self.models[model_repo] = AutoModelForCausalLM.from_pretrained(model_repo)
            self.models[model_repo].eval() # for inference

        text = text.strip()
        if model_repo in self.rephrasers: 
            text = text+" Rephrase:"
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids
        eos_id = self.tokenizer.eos_token_id

        max_length = min(max_length + len(input_ids), 1024)  # Adjusting max_length to account for prompt length

        # Generate text
        outputs = self.models[model_repo].generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            use_cache=True,
            do_sample=False,
            no_repeat_ngram_size=2,
            eos_token_id=eos_id, 
            pad_token_id=eos_id, 
            length_penalty=-1.0,
        )

        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if model_repo in self.rephrasers: 
            new_prompt = re.sub(r'^.*? Rephrase:', '', output_texts[0]).strip()
        else: 
            new_prompt = output_texts[0].strip()

        tokens = clip.tokenize(new_prompt)  # Tokenize the prompt
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)  # Encode the tokens
        return ([[cond, {"pooled_output": pooled}]], new_prompt,)


# Update the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS to include the GPT2PNode
NODE_CLASS_MAPPINGS = {
    "GPT2PNode": GPT2PNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT2PNode": "GPT Prompt Generator"
}
