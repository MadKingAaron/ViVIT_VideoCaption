import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        #print(x is None)
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, frames=64, height=64, width=64, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend_layer = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qvt = nn.Linear(dim, inner_dim * 3, bias=False)
        if not(heads == 1 and dim_head == dim):
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=inner_dim, out_features=dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_layer = nn.Identity()




#######Tester############################### TODO
class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class FactorizedSelfAttention(Attention):
    def __init__(self, dim, frames = 64, height = 64, width = 64, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__(dim, frames, height, width, heads, dim_head, dropout)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        # print("X", x.shape)
        # Get query, key, value
        qkv = self.to_qvt(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # print('Q',q.shape, 'K', k.shape, 'V', v.shape)


        #dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale


        #print(dots.shape)

        attend = self.attend_layer(dots)
        attend = self.dropout(attend)

        # print(attend.shape)
        #output = einsum('b h i j, b h j d -> b h i d', attend, v)
        output = torch.matmul(attend, v)
        output = rearrange(output, 'b h n d -> b n (h d)')

        return self.output_layer(output)

        
    

#### WORK IN PROGRESS ####
class FactorizedDotProductAttention(nn.Module):
    def __init__(self, dim, frames, height, width, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads

        self.frames, self.height, self.width = frames, height, width

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend_layer = nn.Softmax(dim=-1)
        self.query_key_value_layer = nn.Linear(dim, inner_dim * 3, bias=False)

        if not(heads == 1 and dim_head == dim):
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=inner_dim, out_features=dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_layer = nn.Identity()
    
    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        # Get query, key, value

        q, k, v = map(lambda qkv: rearrange(qkv, 'b n (h d) -> b h n d', h=h), self.query_key_value_layer(x).chunk(3, dim=1))

class FFLayer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ff(x)

class FactorizedSelfAttentionTransformerEnc(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, t, h, w, dropout=0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.t, self.h, self.w = t, h, w

        for i in range(depth):
            self.layers.append(self._generate_single_transformer_layer(dim, heads, dim_head, mlp_dim, dropout=dropout))
    
    def _generate_single_transformer_layer(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        layer = nn.ModuleList(
            [
                PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FFLayer(dim, mlp_dim, dropout=dropout))
                
                
            ]
            
        )
        """ # Norm(dim, FactorizedSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), # Spatial SA Block
                Norm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), # Spatial SA Block
                # Norm(dim, FactorizedSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), # Temporal SA Block
                Norm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), # Temporal SA Block
                Norm(dim, FFLayer(dim, mlp_dim, dropout=dropout)) """
        return layer
    
    def _reshape_tensor(self, x:torch.Tensor, b:int, transpose:bool = True, start_dim:int=0, end_dim:int=1):
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        if transpose:
            x = x.transpose(1,2)
        x = torch.flatten(x, start_dim=start_dim, end_dim=end_dim)
        return x

    def forward(self, x):
        b = x.shape[0]
        x = torch.flatten(x, start_dim=0, end_dim=1) # get spacial tokens from x
        # print("X",x.shape)
        count = 0
        for spacial_attn, temp_attn, ff in self.layers:
            spacial_x = spacial_attn(x) + x # Spacial Attn
            # print('sp_attn_layer:', spacial_attn(x).shape)
            


            # Reshape for temporal attn
            spacial_x = self._reshape_tensor(spacial_x, b)

            temp_x = temp_attn(spacial_x) + spacial_x # Temporal Attention

            x = ff(temp_x) + temp_x # MLP

            # Reshape for Spacial Attention
            x = self._reshape_tensor(x, b)

            count += 1
        # Reshape tensor to [b, t*h*w, dim]
        x = self._reshape_tensor(x, b, False, start_dim=1, end_dim=2)
        return x


class ViViT_FSA(nn.Module):
    def __init__(self, frames, frame_size, patch_t, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head=3, channels=3, mode='tubelet', emb_dropout=0.0, dropout=0.0, device=torch.device("cpu"), classifer = False) -> None:
        super().__init__()
        self.T = frames
        self.H, self.W = pair(frame_size)
        self.patch_T = patch_t
        self.patch_H, self.patch_W = pair(patch_size)

        assert self.T % self.patch_T == 0
        assert self.W % self.patch_W == 0
        assert self.H % self.patch_H == 0
        assert self.T % self.patch_T == 0 and self.H % self.patch_H == 0 and self.W % self.patch_W == 0 # Vid dim should be divisible by patch size

        self.mode = mode

        self.nt = self.T // self.patch_T
        self.nh = self.H // self.patch_H
        self.nw = self.W // self.patch_W

        tubelet_dim = self.patch_T * self.patch_H * self.patch_W * channels
        self.tubelet_embedder = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.patch_T, ph=self.patch_H, pw=self.patch_W),
            nn.Linear(tubelet_dim, dim)
        )


        self.pos_embeddings = nn.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to(device)

        self.dropout_layer = nn.Dropout(emb_dropout)
        
        # Only model 3 as of now
        self.transformer = FactorizedSelfAttentionTransformerEnc(dim, depth, heads, dim_head, mlp_dim, self.nt, self.nh, self.nw, dropout)


        self.latent_layer = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Tanh() # To mirror BERT
        )

        self.classifer = classifer
    
    def set_to_classifier(self):
        self.classifer = True
    
    def unset_to_classifer(self):
        self.classifer = False

    def forward(self, x):
        # X is (b, C, T, H, W)

        #print(x.shape)

        tubelets = self.tubelet_embedder(x)
        #print(tubelets.get_device())
        tubelets += self.pos_embeddings
        tubelets = self.dropout_layer(tubelets)

        #print(tubelets.shape)
        x = self.transformer(tubelets)

        #return x
        if self.classifer:
            x = x.mean(dim=1)

            x = self.latent_layer(x)
            output =  self.mlp_head(x)
        else:
            output = x
        
        return output

###################### For testing ############################
if __name__ == '__main__':
    from VideoFrameDataset import VideoFrameDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    x = torch.rand(1, 3, 32, 64, 64).to(device)
    #vivit = ViViT_FSA(32, 64, 64, 8, 4, 4, 10, 512, 6, 10, 8)
    vivit = ViViT_FSA(32, 64, 8, 4, 10, 512, 6, 1000, 8, device=device).to(device)
    print(vivit)
    # vivit.eval()
    # out = vivit(x)
    # print(out.shape)
    # vivit = ViViT_FSA(frames=256, frame_size=512, patch_t=64, patch_size=16, num_classes=10, dim=512, depth=6, heads=500, mlp_dim=4, device=device).to(device)
    # vivit.eval()
    # ds = VideoFrameDataset("EPIC_100_VALIDATION.csv")
    # inp = ds[0].to(device)
    # out = vivit(inp)
