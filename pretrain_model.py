import math
import torch
import vivit
import bert_model
from torch import nn



class PreTrainer():
    def __init__(self, device=torch.device('cpu'), lang_model:str='bert', frames:int=512, frame_size:int=256, patch_t:int=8, patch_size:int=4, dim:int=512, enc_depth:int=6, attn_heads:int=10, mlp_dim:int=8, dropout:float=0.6):
        self.lang_model = bert_model.BERT_Based_Model(device=device, model_type=lang_model)
        self.hidden_size = self.lang_model.get_model_hidden_size()
        self.model = vivit.ViViT_FSA(frames=frames, 
                                frame_size=frame_size, 
                                patch_t=patch_t,
                                patch_size=patch_size,
                                num_classes=self.hidden_size, # TODO: Replace with bert model class size
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
    
    def get_loss(self, training_pred:torch.Tensor, gt:torch.Tensor):
        return self.criterion(training_pred, gt, torch.tensor(1.0))
    
    def backprop_and_step(self, loss, optimizer) -> float:
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def get_batch(self, batch_size:int):
        return None, None
    
    def _print_epoch(self, epoch:int):
        digits = int(math.log10(epoch))+1
        stars = 7
        start = '*'*stars
        end = '*'*(stars - digits)
        print(start+str(epoch)+end)

    def train_model(self, lr:float, epochs:int, batch_size:int):
        optimizer = None
        for epoch in range(epochs):
            self._print_epoch(epoch)
            running_loss = self.train_batch(optimizer, epoch, batch_size)
            print('\nEpoch %d/%d - Running Loss: %.5f\n' %(epoch, epochs, running_loss))
    
    def train_batch(self, optimizer, epoch:int, batch_size:int):
        running_loss = 0.0
        video_clips, captions = self.get_batch(batch_size)

        for i in range(batch_size):
            video_clip = video_clips[i]
            caption = captions[i]
            model_output = self.model(video_clip)
            bert_output = self.lang_model.get_model_CLS_logits(input_string=caption)

            loss = self.get_loss(training_pred=model_output, gt=bert_output)

            current_loss = self.backprop_and_step(loss=loss, optimizer=optimizer)
            running_loss += current_loss

            print("Epoch %d: Batch - %d/%d Current Loss: %.4f, Running Loss: %.4f" %(epoch, i, batch_size, current_loss, running_loss))

        return running_loss
    

if __name__ == '__main__':
    frames = 128
    frame_size = 160
    x = torch.zeros((1, 3, frames, frame_size, frame_size))
    print(x.shape)
    
    trainer = PreTrainer(frames=frames, frame_size=frame_size)
    trainer.model.eval()
    trainer.model.unset_to_classifer()
    print(trainer.model(x).shape)
