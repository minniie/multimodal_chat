from transformers import GPT2Tokenizer, GPT2Model


class ResponseGenerator():

    def __init__(
            self,
            model_name: str = "gpt2"
        ):
        self.model_name = model_name

    def load_tokenizer(
            self
        ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

    def load_model(
            self
        ):
        self.model = GPT2Model.from_pretrained(self.model_name)