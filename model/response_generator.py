from transformers import GPT2Tokenizer, GPT2LMHeadModel


class ResponseGenerator():
    """
    Response generator
    """

    def __init__(
            self,
            generator_model_name: str = "gpt2"
        ):
        self.generator_model_name = generator_model_name
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(
            self
        ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.generator_model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def load_model(
            self
        ):
        self.model = GPT2LMHeadModel.from_pretrained(self.generator_model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))