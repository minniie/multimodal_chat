from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer
) 

from util.text import join_dialog


class ResponseGenerator():
    """
    Response generator
    """

    def __init__(
            self,
            device,
            generator_image_encoder_path: str = None,
            generator_text_decoder_path: str = None,
            generator_finetuned_path: str = None
        ):
        self.device = device
        self.generator_image_encoder_path = generator_image_encoder_path
        self.generator_text_decoder_path = generator_text_decoder_path
        self.generator_finetuned_path = generator_finetuned_path
        self.user_token = "<user>"
        self.bot_token = "<bot>"
        self.set_cls()
        self.load_tokenizer()
        self.load_processor()
        self.load_model()

    def set_cls(
            self
        ):
        self.tokenizer_cls = GPT2Tokenizer
        self.processor_cls = ViTImageProcessor
        if self.generator_image_encoder_path:
            self.model_cls = VisionEncoderDecoderModel
        else:
            self.model_cls = GPT2LMHeadModel
    
    def load_tokenizer(
            self
        ):
        if self.generator_finetuned_path:
            self.tokenizer = self.tokenizer_cls.from_pretrained(self.generator_finetuned_path)
        else:
            self.tokenizer = self.tokenizer_cls.from_pretrained(self.generator_text_decoder_path)
            special_tokens = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}
            self.tokenizer.add_special_tokens(special_tokens)
            self.tokenizer.add_tokens([self.user_token, self.bot_token])
        
    def load_processor(
            self
        ):
        if self.generator_image_encoder_path:
            self.processor = self.processor_cls.from_pretrained(self.generator_image_encoder_path)
        else:
            self.processor = None

    def load_model(
            self
        ):
        # loading finetuned checkpoint
        if self.generator_finetuned_path:
            self.model = self.model_cls.from_pretrained(self.generator_finetuned_path)
            self.model.to(self.device)

        # loading pretrained checkpoint from huggingface
        else:        
            if self.generator_image_encoder_path:
                self.model = self.model_cls.from_encoder_decoder_pretrained(
                    self.generator_image_encoder_path, self.generator_text_decoder_path
                )
                self.model.to(self.device)
                self.model.decoder.resize_token_embeddings(len(self.tokenizer))
                self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.config.bos_token_id = self.tokenizer.bos_token_id
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.vocab_size = self.model.config.decoder.vocab_size
            else:
                self.model = self.model_cls.from_pretrained(self.generator_text_decoder_path)
                self.model.to(self.device)
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