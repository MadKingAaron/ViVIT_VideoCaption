import vivit
import modified_vivit
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


class ViVIT_CaptionerModel3(nn.Module):
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

class ViVIT_CaptionerModel2(nn.Module):
    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_p=0.,
                 tube_size=2,
                 conv_type='Conv3d',
                 attention_type='fact_encoder',
                 norm_layer=nn.LayerNorm,
                 return_cls_token=True,
                 dec_heads = 8,
                 dec_depth = 6,
                 **kwargs) -> None:
        super().__init__()
        self.encoder = modified_vivit.ViViTModel2(num_frames,
                 img_size,
                 patch_size,
                 embed_dims,
                 num_heads,
                 num_transformer_layers,
                 in_channels,
                 dropout_p,
                 tube_size,
                 conv_type,
                 attention_type,
                 norm_layer,
                 return_cls_token)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=dec_heads)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=dec_depth)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get image embedding
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

    vivit = ViVIT_CaptionerModel3(frames=32, 
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

