import torch
import math
import re
import numpy as np

import warnings
#from utils import print_on_rank_zero

from einops import rearrange, reduce, repeat

from torch import nn

from transformer_components import TransformerContainer

from pytorch_lightning.utilities.distributed import rank_zero_only
import torch.distributed as dist


@rank_zero_only
def print_on_rank_zero(content):
	if is_main_process():	
		print(content)

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True

def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()

def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()

def is_main_process():
	return get_rank() == 0



def _pair(x):
	if isinstance(x, tuple):
		return x
	else:
		return x, x


def get_sine_cosine_pos_emb(n_position, d_hid): 
	''' Sinusoid position encoding table ''' 
	# TODO: make it with torch instead of numpy 
	def get_position_angle_vec(position): 
		return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

	sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)

########## Taken from mx-mark (https://github.com/mx-mark/VideoTransformer-pytorch) ##########

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

@torch.no_grad()	
def init_from_vit_pretrain_(module,
						    pretrained,
						    conv_type, 
						    attention_type, 
						    copy_strategy,
						    extend_strategy='temporal_avg',
						    tube_size=2, 
						    num_time_transformer_layers=4):

	if isinstance(pretrained, str):
		if torch.cuda.is_available():
			state_dict = torch.load(pretrained)
		else:
			state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
		
		if 'state_dict' in state_dict:
			state_dict = state_dict['state_dict']
		
		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			# extend the Conv2d params to Conv3d
			if conv_type == 'Conv3d':
				if 'patch_embed.projection.weight' in old_key:
					weight = state_dict[old_key]
					print("*"*24, weight.shape)
					new_weight = repeat(weight, 'd c h w -> d c t h w', t=tube_size)
					if extend_strategy == 'temporal_avg':
						new_weight = new_weight / tube_size
					elif extend_strategy == 'center_frame':
						new_weight.zero_()
						new_weight[:,:,tube_size//2,:,:] = weight
					state_dict[old_key] = new_weight
					continue
					
			# modify the key names of norm layers
			if attention_type == 'fact_encoder':
				new_key = old_key.replace('transformer_layers.layers',
										  'transformer_layers.0.layers')
			else:
				new_key = old_key
			
			if 'in_proj' in new_key:
				new_key = new_key.replace('in_proj_', 'qkv.') #in_proj_weight -> qkv.weight
			elif 'out_proj' in new_key:
				new_key = new_key.replace('out_proj', 'proj')

			if 'norms' in new_key:
				new_key = new_key.replace('norms.0', 'attentions.0.norm')
				new_key = new_key.replace('norms.1', 'ffns.0.norm')

			state_dict[new_key] = state_dict.pop(old_key)

		old_state_dict_keys = list(state_dict.keys())
		for old_key in old_state_dict_keys:
			# copy the parameters of space attention to time attention
			if attention_type == 'divided_space_time':
				if 'attentions.0' in old_key:
					new_key = old_key.replace('attentions.0',
											  'attentions.1')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()
			# copy the part of parameters of space attention to time attention
			elif attention_type == 'fact_encoder':
				pattern = re.compile(r'(?<=layers.)\d+')
				matchObj = pattern.findall(old_key)
				if len(matchObj) > 1 and int(matchObj[1]) < num_time_transformer_layers:
					new_key = old_key.replace('transformer_layers.0.layers', 
											  'transformer_layers.1.layers')
					if copy_strategy == 'repeat':
						state_dict[new_key] = state_dict[old_key].clone()
					elif copy_strategy == 'set_zero':
						state_dict[new_key] = state_dict[old_key].clone().zero_()

		missing_keys,unexpected_keys = module.load_state_dict(state_dict, strict=False)
		#print(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')
		print_on_rank_zero(f'missing_keys:{missing_keys}\n '
						   f'unexpected_keys:{unexpected_keys}')


def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:] # skip 'model.'
			if 'in_proj' in new_key:
				new_key = new_key.replace('in_proj_', 'qkv.') #in_proj_weight -> qkv.weight
			elif 'out_proj' in new_key:
				new_key = new_key.replace('out_proj', 'proj')
			state_dict[new_key] = state_dict.pop(old_key)
		else: # cls_head
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)

def init_from_kinetics_pretrain_(module, pretrain_pth):
	if torch.cuda.is_available():
		state_dict = torch.load(pretrain_pth)
	else:
		state_dict = torch.load(pretrain_pth, map_location=torch.device('cpu'))
	if 'state_dict' in state_dict:
		state_dict = state_dict['state_dict']
	
	replace_state_dict(state_dict)
	msg = module.load_state_dict(state_dict, strict=False)
	print_on_rank_zero(msg)


@torch.no_grad()
def kaiming_init_(tensor,a=0, mode='fan_out', nonlinearity='relu', distribution='normal'):
	assert distribution in ['uniform', 'normal']
	if distribution == 'uniform':
		nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
	else:
		nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

@torch.no_grad()   
def constant_init_(tensor, constant_value=0):
	nn.init.constant_(tensor, constant_value)

class PatchEmbed(nn.Module):
	"""Images to Patch Embedding.
	Args:
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		tube_size (int): Size of temporal field of one 3D patch.
		in_channels (int): Channel num of input features. Defaults to 3.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
	"""

	def __init__(self,
				 img_size,
				 patch_size,
				 tube_size=2,
				 in_channels=3,
				 embed_dims=768,
				 conv_type='Conv2d'):
		super().__init__()
		self.img_size = _pair(img_size)
		self.patch_size = _pair(patch_size)

		num_patches = \
			(self.img_size[1] // self.patch_size[1]) * \
			(self.img_size[0] // self.patch_size[0])
		assert (num_patches * self.patch_size[0] * self.patch_size[1] == 
			   self.img_size[0] * self.img_size[1],
			   'The image size H*W must be divisible by patch size')
		self.num_patches = num_patches

		# Use conv layer to embed
		if conv_type == 'Conv2d':
			self.projection = nn.Conv2d(
				in_channels,
				embed_dims,
				kernel_size=patch_size,
				stride=patch_size)
		elif conv_type == 'Conv3d':
			self.projection = nn.Conv3d(
				in_channels,
				embed_dims,
				kernel_size=(tube_size,patch_size,patch_size),
				stride=(tube_size,patch_size,patch_size))
		else:
			raise TypeError(f'Unsupported conv layer type {conv_type}')
			
		self.init_weights(self.projection)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		layer_type = type(self.projection)
		if layer_type == nn.Conv3d:
			x = rearrange(x, 'b t c h w -> b c t h w')
			x = self.projection(x)
			x = rearrange(x, 'b c t h w -> (b t) (h w) c')
		elif layer_type == nn.Conv2d:
			x = rearrange(x, 'b t c h w -> (b t) c h w')
			x = self.projection(x)
			x = rearrange(x, 'b c h w -> b (h w) c')
		else:
			raise TypeError(f'Unsupported conv layer type {layer_type}')
		
		return x


class ViViTModel3(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>
	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size=224,
				 patch_size=16,
				 pretrain_pth=None,
				 weights_from='imagenet',
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 dropout_p=0.,
				 tube_size=2,
				 conv_type='Conv3d',
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')
		
		num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.pretrain_pth = pretrain_pth
		self.weights_from = weights_from
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.conv_type = conv_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			tube_size=tube_size,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrain_pth is not None:
			if self.weights_from == 'imagenet':
				init_from_vit_pretrain_(self,
										self.pretrain_pth,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			elif self.weights_from == 'kinetics':
				init_from_kinetics_pretrain_(self,
											 self.pretrain_pth)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def prepare_tokens(self, x):
		#Tokenize
		b = x.shape[0]
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, cls_tokens, b

	def forward(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_last_selfattention(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, return_attention=True)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			print(x.shape)
			x = temporal_transformer(x, return_attention=True)
		return x
	
class ViViTModel2(nn.Module):
    """ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
        

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to 12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_p (float): Probability of dropout layer. Defaults to 0..
        tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
        conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
            Defaults to 'fact_encoder'.
        norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
        copy_strategy (str): Copy or Initial to zero towards the new additional layer.
        extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
        use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
        return_cls_token (bool): Whether to use cls_token to predict class label.
    """
    supported_attention_types = [
        'fact_encoder', 'joint_space_time', 'divided_space_time'
    ]

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
                 **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')

        num_frames = num_frames//tube_size
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.conv_type = conv_type
        self.tube_size = tube_size
        self.num_time_transformer_layers = 4
        self.return_cls_token = return_cls_token

        #tokenize & position embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            tube_size=tube_size,
            conv_type=conv_type)
        num_patches = self.patch_embed.num_patches

        # Divided Space Time Transformer Encoder - Model 2
        transformer_layers = nn.ModuleList([])

        spatial_transformer = TransformerContainer(
            num_transformer_layers=num_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        temporal_transformer = TransformerContainer(
            num_transformer_layers=self.num_time_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims*4,
            operator_order=['self_attn','ffn'])

        transformer_layers.append(spatial_transformer)
        transformer_layers.append(temporal_transformer)

        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
        # whether to add one cls_token in temporal pos_enb
        num_frames = num_frames + 1
        num_patches = num_patches + 1

        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
        self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        #Tokenize
        b, t, c, h, w = x.shape
        x = self.patch_embed(x)

        # Add Position Embedding
        cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.drop_after_pos(x)

        # fact encoder - CRNN style
        spatial_transformer, temporal_transformer, = *self.transformer_layers,
        print(x.shape)
        x = spatial_transformer(x)

        print(x.shape)
        # Add Time Embedding
        cls_tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
        x = reduce(x, 'b t p d -> b t d', 'mean')
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)
        print('Time:',self.time_embed.shape)
        x = x + self.time_embed
        x = self.drop_after_time(x)

        x = temporal_transformer(x)

        x = self.norm(x)
        return x
        # Return Class Token
        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)

def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:]
			state_dict[new_key] = state_dict.pop(old_key)
		else:
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)

def load_pretrained(model:nn.Module, weight_dir:str):
	state_dict = torch.load(weight_dir)

	replace_state_dict(state_dict)

	model.load_state_dict(state_dict, strict=False)

def create_lookahead_mask(shape:int) -> torch.Tensor:
    return torch.triu(torch.ones(shape, shape) * float('-inf'), diagonal=1)
if __name__ == "__main__":
	num_frames = 8 * 2
	img_size = 224
	inp = torch.zeros(1, num_frames, 3, img_size, img_size)
	model = ViViTModel2(num_frames=num_frames,
                img_size=img_size,
                patch_size=16,
                embed_dims=768,
                in_channels=3,
                attention_type='fact_encoder',
                return_cls_token=False,
				conv_type='Conv3d')
	load_pretrained(model, './vivit_model.pth')
	outpt = model(inp)
	print(torch.squeeze(outpt).shape)
	print(outpt[0][0].shape)
	print(outpt[:,0][0].shape)
	print(torch.equal(outpt[0][0], outpt[:,0][0]))
	decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
	decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=4)
	mask = create_lookahead_mask(54)
	tgt = torch.rand(54, 768)
	print(decoder(tgt, outpt[0], tgt_mask=mask).shape)
	#print(model(inp)[:].shape)