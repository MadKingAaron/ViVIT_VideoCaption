import vivit
import torch
from torch import nn

def generate_tgt(rows:int, columns:int, start_index:int = 12) -> torch.Tensor:
    tgt = torch.zeros(rows, columns)
    tgt[start_index][0] = 1
    return tgt

def create_lookahead_mask(shape:int) -> torch.Tensor:
    return torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)


def get_loss(criterion, outputs:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
    loss = criterion(torch.t(outputs), labels)
    return loss


class ViVIT_Captioner(nn.Module):
    def __init__(self, frames, frame_size, patch_t, patch_size, num_classes, dim, enc_depth, dec_depth, enc_heads=1000, enc_mlp_dim=8, dec_heads=8, enc_dim_head=3, channels=3, mode='tubelet', emb_dropout=0, dropout=0, device=torch.device("cpu")) -> None:
        super().__init__()
        self.encoder = vivit.ViViT_FSA(frames, frame_size, patch_t, patch_size, num_classes, dim, enc_depth, enc_heads, enc_mlp_dim, enc_dim_head, channels, mode, emb_dropout, dropout, device=device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=dec_heads)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=dec_depth)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        image_embedding = self.encoder(x)
        
        # Get the image embedding for 2-D
        memory = image_embedding[0]

        # Decoder
        rows, columns = memory.shape
        tgt = generate_tgt(rows, columns) # TODO: ADD START INDEX
        mask = create_lookahead_mask(rows)

        out = self.decoder(tgt, memory, tgt_mask=mask)

        return out

if __name__ == "__main__":
    device = torch.device('cpu')
    x = torch.rand(1, 3, 32, 64, 64).to(device)

    vivit = ViVIT_Captioner(frames=32, 
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
    
    vivit.eval()
    out = vivit(x)
    print(out.shape)
    print(vivit)

