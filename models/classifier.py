from .model import *
from .pretrainmodel import SAINT_encoder
#from einops import rearrange

class classifier(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        dim,
        y_dim = 2,
        dropout = 0.
        ):
        super().__init__()
        
        self.encoder = encoder

        #total_dim = (encoder.num_categories+encoder.num_continuous)*dim
        self.mlpfory = MLP_dropout([dim, 256, 128, 64, y_dim],dropout=dropout)
        self.mlpphi = MLP_dropout([dim, 4*dim, 4*dim, 2*dim, dim],dropout=dropout)
        self.dim = dim
        
        self.categories_offset = encoder.categories_offset
        self.cat_mask_offset = encoder.cat_mask_offset
        self.con_mask_offset = encoder.con_mask_offset
        self.mask_embeds_cat = encoder.mask_embeds_cat
        self.mask_embeds_cont = encoder.mask_embeds_cont
        self.embeds = encoder.embeds
        self.cont_embeddings = encoder.cont_embeddings
        self.num_continuous = encoder.num_continuous
        self.simple_MLP = encoder.simple_MLP       

    def forward(self, x_categ, x_cont):
        x = self.encoder.transformer(x_categ, x_cont)
        #x = rearrange(x, 'b n d -> b (n d)')
        x = x[:,0,:]
        x = self.mlpphi(x)
        x = x.sum(dim=0)
        x = x.reshape(1, -1)
        x = self.mlpfory(x)
        return x
