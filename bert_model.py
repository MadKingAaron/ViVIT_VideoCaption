import transformers
from transformers import AutoTokenizer, BertModel, RobertaModel, BertConfig, RobertaConfig
import torch
from types import FunctionType





class BERT_Based_Model():
    def __init__(self, device = torch.device('cpu'), model_type:str = 'bert') -> None:
        if model_type == 'bert':
            self.model, self.tokenizer, self.config = self.get_bert_model()
        elif model_type == 'roberta':
            self.model, self.tokenizer, self.config = self.get_roberta_model()
        else: # Default to BERT, if unknown model type
            self.model, self.tokenizer, self.config = self.get_bert_model()
        
        self.model = self.model.to(device)
    
    def _get_model(self, model_creator:FunctionType = BertModel.from_pretrained, config_func:FunctionType = BertConfig.from_pretrained,model_name:str = 'bert-large-uncased'):
        tokenzier = AutoTokenizer.from_pretrained(model_name)
        model = model_creator(model_name)
        config = config_func(model_name)

        return model, tokenzier, config

    def get_bert_model(self):
        return self._get_model(BertModel.from_pretrained, BertConfig.from_pretrained, 'bert-large-uncased')

    def get_roberta_model(self):
        return self._get_model(RobertaModel.from_pretrained, RobertaConfig.from_pretrained, 'roberta-base')
    
    def _get_model_outputs(self, input_string:str, max_length:int = 512):
        inputs = self.tokenizer(input_string, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
        outputs = self.model(**inputs)

        return outputs

    def get_model_CLS_logits(self, input_string:str, max_length:int = 512) -> torch.Tensor:
        return self._get_model_outputs(input_string, max_length)[1]
    
    def get_model_hidden_size(self) -> int:
        return self.config.hidden_size
