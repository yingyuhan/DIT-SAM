import torch
from torch import nn
from models.module_lib import Adapter


class SAMImageEncodeWrapper(nn.Module):  # 冻结参数

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMImageEncodeWrapper, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x


class SAMPromptEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMPromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()


class DITSAMImageEncoder(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,
    ):
        super(DITSAMImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)
        self.hq_token = hq_token 

        total_p_layer = len(self.sam_img_encoder.blocks) 
        prompt_dim = self.sam_img_encoder.pos_embed.shape[-1]   
        self.hq_token_proj = nn.Sequential(

             *[Adapter() for _ in range(total_p_layer)]
        )   


    def forward(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        hq_prompt_tokens = []
        for i in range(0, len(self.hq_token_proj)):
            hq_prompt_tokens.append(self.hq_token_proj[i](self.hq_token).unsqueeze(0))  # torch.Size([1, 1, 1024])

        interm_embeddings = []
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x, hq_prompt_tokens[i])   

            if blk.window_size == 0:  
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings   


