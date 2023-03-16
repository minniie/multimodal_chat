from transformers import (
    BlipForQuestionAnswering,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModel,
    AutoTokenizer,
    AutoProcessor
) 

from util.text import join_dialog


class ResponseGenerator():
    """
    Response generator
    """

    def __init__(
            self,
            device,
            generator_model_name_or_path: str = "gpt2",
            use_image_as_generator_input: bool = False
        ):
        self.device = device
        self.generator_model_name_or_path = generator_model_name_or_path
        self.use_image_as_generator_input = use_image_as_generator_input
        self.user_token = "<user>"
        self.bot_token = "<bot>"
        self.set_cls()
        self.load_tokenizer()
        self.load_processor()
        self.load_model()

    def set_cls(
            self
        ):
        if "blip" in self.generator_model_name_or_path.lower():
            self.tokenizer_cls, self.model_cls = AutoTokenizer, BlipForQuestionAnswering
        elif "gpt" in self.generator_model_name_or_path.lower():
            self.tokenizer_cls, self.model_cls = GPT2Tokenizer, GPT2LMHeadModel
        else:
            self.tokenizer_cls, self.model_cls = AutoTokenizer, AutoModel
        self.processor_cls = AutoProcessor
    
    def load_tokenizer(
            self
        ):
        # self.tokenizer = self.tokenizer_cls.from_pretrained(self.generator_model_name_or_path)
        self.tokenizer = self.tokenizer_cls.from_pretrained("gpt2-medium")
        # if not self.tokenizer.sep_token:
        #     self.tokenizer.sep_token = "\n"
        # if not self.tokenizer.pad_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        if "gpt" in self.generator_model_name_or_path.lower():
            self.tokenizer.add_special_tokens({
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>"
            })
            self.tokenizer.add_tokens([self.user_token, self.bot_token])

    def load_processor(
            self
        ):
        self.processor = None
        if self.use_image_as_generator_input:
            self.processor = self.processor_cls.from_pretrained(self.generator_model_name_or_path)

    def load_model(
            self
        ):
        self.model = self.model_cls.from_pretrained(self.generator_model_name_or_path)
        self.model.to(self.device)
        if "gpt" in self.generator_model_name_or_path.lower():
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def inference(
            self,
            context
        ):
        inp_text = join_dialog(context, self.tokenizer.sep_token) + self.tokenizer.sep_token
        inp = self.tokenizer.encode(inp_text, return_tensors="pt").to(self.model.device)
        pred = self.model.generate(inp, max_length=64, num_beams=1, do_sample=True)
        pred_text = self.tokenizer.decode(pred[0][:-1])
        pred_text = pred_text.replace(inp_text, "")
        if not pred_text:
            pred_text = "[END OF DIALOGUE]"

        return pred_text