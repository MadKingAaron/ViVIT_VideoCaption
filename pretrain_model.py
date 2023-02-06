import math
import torch
import vivit
import bert_model
import VideoFrameDataset
from torch import nn
from tensorboard import TensorBoard

from sys import platform


class NoDatasetError(Exception):
    def __init__(self) -> None:
        self.message = "No dataset has been created yet. Use 'setDataset'"
        super().__init__(self.message)


def set_device():
    if platform == "darwin": # If Mac OS
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif platform == 'linux' or platform == 'win32': # If Linux or Windows
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device('cpu')

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

        self.total_frames = frames
        self.frame_size = frame_size
        self.dataset = None

    def set_dataset(self, dataset_csv:str, video_file_col:str = 'video_file', video_folder_dir:str = './pretrain_data'):
        self.dataset = VideoFrameDataset.PretrainDataset(dataset_csv=dataset_csv, frame_shape=self.frame_size, total_frames=self.total_frames, 
                    video_file_col=video_file_col, video_folder_dir=video_folder_dir)

    def get_lang_model_output(self, str_input:str) -> torch.Tensor:
        return self.lang_model.get_model_CLS_logits(input_string=str_input)
    
    def get_loss(self, training_pred:torch.Tensor, gt:torch.Tensor):
        return self.criterion(training_pred, gt, torch.tensor(1.0))
    
    def backprop_and_step(self, loss, optimizer) -> float:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def get_batch(self, batch_size:int=256) -> tuple[torch.Tensor, list]:
        self.dataset_check()
        return self.dataset.get_batch(batch_size)
    
    def _print_epoch(self, epoch:int):
        digits = int(math.log10(epoch))+1
        stars = 7
        start = '*'*stars
        end = '*'*(stars - digits)
        print(start+str(epoch)+end)

    def dataset_check(self):
        if self.dataset is None:
            raise NoDatasetError()
        
    def train_model(self, lr:float, epochs:int, batch_size:int):
        self.dataset_check()

        optimizer = None # TODO: Need to set optimizer
        for epoch in range(epochs):
            self._print_epoch(epoch)
            running_loss = self.train_batch(optimizer, epoch, batch_size)
            print('\nEpoch %d/%d - Running Loss: %.5f\n' %(epoch, epochs, running_loss))
    
    def train_batch(self, optimizer, epoch:int, batch_size:int):
        running_loss = 0.0
        video_clips, captions = self.get_batch(batch_size)

        for i in range(batch_size):

            # optimizer.zero_grad()

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
