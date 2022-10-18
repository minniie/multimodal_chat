from transformers import GPT2Tokenizer, GPT2LMHeadModel

from util.text import join_dialog


class ResponseGenerator():
    """
    Response generator
    """

    def __init__(
            self,
            device,
            generator_model_name_or_path: str = "gpt2"
        ):
        self.device = device
        self.generator_model_name_or_path = generator_model_name_or_path
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(
            self
        ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.generator_model_name_or_path)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def load_model(
            self
        ):
        self.model = GPT2LMHeadModel.from_pretrained(self.generator_model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
    
    def inference(
            self,
            context
        ):
        inp_text = join_dialog(context, self.tokenizer.eos_token) + self.tokenizer.eos_token
        inp = self.tokenizer.encode(inp_text, return_tensors="pt").to(self.model.device)
        pred = self.model.generate(inp, max_length=64, num_beams=1, do_sample=True)
        pred_text = self.tokenizer.decode(pred[0][:-1])
        pred_text = pred_text.replace(inp_text, "")
        if not pred_text:
            pred_text = "[END OF DIALOGUE]"
        
        print("======= input text =======")
        print(inp_text)
        print("======= pred text =======")
        print(pred_text)
        print("======= pred text original =======")
        print(self.tokenizer.decode(pred[0]))

        return pred_text