import transformers
from transformers import AutoTokenizer, BertModel, RobertaModel
import torch
from types import FunctionType





class BERT_Based_Model():
    def __init__(self, model_type:str = 'bert') -> None:
        if model_type == 'bert':
            self.model, self.tokenizer = self.get_bert_model()
        elif model_type == 'roberta':
            self.model, self.tokenizer = self.get_roberta_model()
        else: # Default to BERT, if unknown model type
            self.model, self.tokenizer = self.get_bert_model()
    
    def _get_model(self, model_creator:FunctionType = BertModel.from_pretrained, model_name:str = 'bert-large-uncased'):
        tokenzier = AutoTokenizer.from_pretrained(model_name)
        model = model_creator(model_name)

        return model, tokenzier

    def get_bert_model(self):
        return self._get_model(BertModel.from_pretrained, 'bert-large-uncased')

    def get_roberta_model(self):
        return self._get_model(RobertaModel.from_pretrained, 'roberta-base')
    
    def _get_model_outputs(self, input_string:str, max_length:int = 512):
        inputs = self.tokenizer(input_string, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        outputs = self.model(**inputs)

        return outputs

    def get_model_CLS_logits(self, input_string:str, max_length:int = 512) -> torch.Tensor:
        return self._get_model_outputs(input_string, max_length)[1]

