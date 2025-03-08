import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
import math

class BootstrappedMAE(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.teacher_model = featmae_vit_tiny_patch4_dec5128b()  
        self.student_model = featmae_vit_tiny_patch4_dec5128b()
        self.update_teacher()

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.ema_decay = args.ema_decay_init

        self.LN = nn.LayerNorm(self.student_model.embed_dim, eps=1e-6, elementwise_affine=False).cuda()

    def forward(self,imgs,mask_ratio=0.75):
        latent_student, mask, ids_restore = self.student_model.forward_encoder(imgs, mask_ratio)    # latent:[N,1+(unmasked patch num),C]
        pred_student = self.student_model.forward_decoder(latent_student,ids_restore)   # [N,L,C]

        latent_teacher,_,_ = self.teacher_model.forward_encoder(imgs,0.0)

        loss = self.forward_loss(pred_student,latent_teacher[:,1:,:],mask)

        return loss, pred_student, mask

    def forward_loss(self,pred_student,latent_teacher,mask):
        # pred_student: [N,L,C] 
        # latent_teacher: [N,L,C]
        # mask: [N,L], 0 is keep, 1 is remove

        loss = (self.LN(pred_student) - self.LN(latent_teacher)) ** 2
        loss = loss.mean(dim=-1)    # [N,L]

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            

        return loss
    def update_decay_cosine(self,epoch,args):
        # 余弦衰减ema的参数，使开始时更新快，后期更新慢
        if epoch<args.ema_decay_warmup_epoch:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / args.ema_decay_warmup_epoch))
            ema_decay = args.ema_decay_final + (args.ema_decay_init - args.ema_decay_final) * cosine_decay
            self.ema_decay = ema_decay
        else:
            self.ema_decay = args.ema_decay_final

    @torch.no_grad()
    def update_teacher(self,method='copy'):
        # method = copy or ema
        if method == 'copy':
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data.copy_(student_param.data)
        elif method == 'ema':
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
    
class FeatureMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()


        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to encoder's latent dim
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.embed_dim = embed_dim


def featmae_vit_tiny_patch4_dec5128b(**kwargs):
    model = FeatureMaskedAutoencoderViT(
        img_size=32,
        patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

