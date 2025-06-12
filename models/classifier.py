from .model import *
from .pretrainmodel import SAINT_encoder

class classifier(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        dim,
        y_dim = 2
        ):
        super().__init__()

        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.encoder = encoder
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
        x = x[:,0,:]
        x = x.sum(dim=0)
        x = x.reshape(1, -1)
        x = self.mlpfory(x)
        return x
