import torch
import vivit
import bert_model
from torch import nn


class Trainer():
    def __init__(self, device=torch.device('cpu'), lang_model:str='bert', frames:int=512, frame_size:int=256, patch_t:int=8, patch_size:int=4, dim:int=512, enc_depth:int=6, attn_heads:int=10, mlp_dim:int=8, dropout:float=0.0):
        self.lang_model = bert_model.BERT_Based_Model(device=device, model_type=lang_model)
        self.model = vivit.ViViT_FSA(frames=frames, 
                                frame_size=frame_size, 
                                patch_t=patch_t,
                                patch_size=patch_size,
                                num_classes=1024, # TODO: Replace with bert model class size
                                dim=dim,
                                depth=enc_depth,
                                heads=attn_heads,
                                mlp_dim=mlp_dim,
                                emb_dropout=dropout,
                                dropout=dropout,
                                device=device,
                                classifer=True).to(device)
        
        self.criterion = nn.CosineEmbeddingLoss()

    def get_lang_model_output(self, str_input:str) -> torch.Tensor:
        return self.lang_model.get_model_CLS_logits(input_string=str_input)
    
    

if __name__ == '__main__':
    frames = 128
    frame_size = 160
    x = torch.zeros((1, 3, frames, frame_size, frame_size))
    print(x.shape)
    
    trainer = Trainer(frames=frames, frame_size=frame_size)
    trainer.model.eval()
    trainer.model.unset_to_classifer()
    print(trainer.model(x).shape)
