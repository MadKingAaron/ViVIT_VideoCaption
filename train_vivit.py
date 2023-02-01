import VideoFrameDataset
import torch
from vivit_captioner import ViVIT_Captioner
import transformers
from tensorboard import TensorBoard

from sys import platform

def set_device():
    if platform == "darwin": # If Mac OS
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif platform == 'linux' or platform == 'win32': # If Linux or Windows
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device('cpu')

device = set_device()

class ViVitTrainer():
    def __init__(self, dataset_csv:str, checkpt_freq:int=5) -> None:
        self.ds = VideoFrameDataset.VideoFrameDataset(dataset_csv)
        self.model =  ViVIT_Captioner(frames=32, 
                            frame_size=64, 
                            patch_t=8, 
                            patch_size=4, 
                            num_classes=10,
                            dim=512,
                            enc_depth=6,
                            dec_depth=6,
                            enc_heads=1000,
                            device=device
                            )
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.checkpt_freq = checkpt_freq
        self.TB = TensorBoard()
    
    def train(self, epochs = 100, lr = 0.001, batch_size = 256):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            epoch_loss = self._train_epoch(epoch, criterion, optimizer, batch_size)
            print('Epoch:\t%d/%d\tEpoch total loss: %.4f' %(epoch, epochs, epoch_loss))

            # Update TensorBoard
            self.TB.write_train_loss(epoch_loss, epoch)

            # Save checkpoint
            if epoch % self.checkpt_freq == 0:
                self.save_model_checkpoint(epoch)
        
        # Save final model
        self.save_model()

    def save_model_checkpoint(self, epoch:int):
        name = 'checkpoint-'+str(epoch)
        self.save_model(name)
    
    def save_model(self, name='final'):
        full_path = './models_saves/vivit_caption_'+name+".pth"
        torch.save(self.model.state_dict(), full_path)


    def _train_epoch(self, epoch, criterion, optimizer, batch_size = 256):
        trained = 0
        running_loss = 0.0
        for frames, gt_caption in self.ds.get_random_batch(batch_size):
            gt_caption_enc = self._encode_text(gt_caption)

            optimizer.zero_grad()

            output = self.model(frames)
            loss = criterion(output, gt_caption_enc)
            loss.backward()
            optimizer.step()
            
            trained += 1
            running_loss += loss.item()
        
        return running_loss
    
    def load_model(self, path:str):
        self.model.load_state_dict(torch.load(path))

    def val(self, batch_size:int = 256):
        self.model.eval()
    
    def test(self):
        self.model.eval()

    
    def _encode_text(self, text:str, max_length:int=512, truncation:bool=True) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, padding="max_length", max_length=max_length, truncation=truncation))