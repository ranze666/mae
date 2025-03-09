import torch
import torch.nn as nn
from models_mae import MaskedAutoencoderViT
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
import math
from util.pos_embed import get_2d_sincos_pos_embed

class BootstrappedMAE(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.teacher_model = featmae_vit_tiny_patch4_dec5128b(args=args)  
        self.student_model = featmae_vit_tiny_patch4_dec5128b(args=args)
        self.update_teacher()

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.ema_decay = args.ema_decay_init

        self.LN = nn.LayerNorm(self.student_model.embed_dim, eps=1e-6, elementwise_affine=False).cuda() if args.ln_feature else nn.Identity()
        self.args = args
        self.pixel_alpha = 1

    def forward(self,imgs,mask_ratio=0.75):
        latent_student_for_pixel, mask, ids_restore, latent_student_for_feat = self.student_model.forward_encoder(imgs, mask_ratio)    # latent:[N,1+(unmasked patch num),C]

        _,_,_ ,latent_teacher_for_feat= self.teacher_model.forward_encoder(imgs,0.0)


        # 重建feature
        pred_student_feature = self.student_model.forward_feature_decoder(latent_student_for_feat,ids_restore)   # [N,L,C]
        

        # 重建pixel,始终使用最后一层的输出
        if self.args.use_pixel:
            pred_student_pixel = self.student_model.forward_pixel_decoder(latent_student_for_pixel,ids_restore)   # [N,L,C]
        else:
            pred_student_pixel = None
        

        loss = self.forward_loss(imgs,pred_student_pixel,pred_student_feature,latent_teacher_for_feat[:,1:,:],mask)

        return loss, None, None

    def forward_loss(self,imgs,pred_student_pixel,pred_student_feature,latent_teacher,mask):
        # pred_student: [N,L,C] 
        # latent_teacher: [N,L,C]
        # mask: [N,L], 0 is keep, 1 is remove

        feature_loss = (self.LN(pred_student_feature) - self.LN(latent_teacher)) ** 2
        feature_loss = feature_loss.mean(dim=-1)    # [N,L]
        feature_loss = (feature_loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if self.args.use_pixel:
            target = self.student_model.patchify(imgs)
            if self.student_model.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
            pixel_loss = (pred_student_pixel - target) ** 2
            pixel_loss = pixel_loss.mean(dim=-1)    # [N,L]
            pixel_loss = (pixel_loss * mask).sum() / mask.sum()  # mean loss on removed patches

            loss = feature_loss*(1-self.pixel_alpha) + pixel_loss*self.pixel_alpha
        else:
            loss = feature_loss



        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))

        return loss
    

    def update_decay_cosine(self,epoch,args):
        # 余弦衰减ema的参数，使开始时更新快，后期更新慢
        if epoch<args.ema_decay_warmup_epoch:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / args.ema_decay_warmup_epoch))
            ema_decay = args.ema_decay_final + (args.ema_decay_init - args.ema_decay_final) * cosine_decay
            self.ema_decay = ema_decay
            
            if self.args.pixel_loss_decay:
            # 顺便在这里把pixel的权重也衰减了
                self.pixel_alpha = cosine_decay
            else:
                self.pixel_alpha = 0.5
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
    
class FeatureMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False,args=None):
        super().__init__()

        self.args = args

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.pixel_norm = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)   
        
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder for feature reconstruction
        self.feature_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.feature_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.feature_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.feature_decoder_norm = norm_layer(decoder_embed_dim)
        self.feature_decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to encoder's latent dim
        # --------------------------------------------------------------------------

        if args.use_pixel:
            # --------------------------------------------------------------------------
            # decoder for pixel reconstruction
            self.pixel_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.pixel_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.pixel_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.pixel_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.pixel_decoder_norm = norm_layer(decoder_embed_dim)
            self.pixel_decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            # --------------------------------------------------------------------------

            self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.embed_dim = embed_dim

    def forward_encoder(self, x, mask_ratio,):
        # 默认使用最后一层(11层)的特征来重建
        # embed patches
        x = self.patch_embed(x)     # n,3,h,w -> n,(h*w)/(ps*ps), c

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)   # n,16?,192

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


         
        # apply Transformer blocks
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.args.feature_layer:
                feature_out = self.norm(x)
        x = self.pixel_norm(x)

        return x, mask, ids_restore, feature_out
    
    def forward_pixel_decoder(self, x, ids_restore):
        # embed tokens
        x = self.pixel_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.pixel_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pixel_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.pixel_decoder_blocks:
            x = blk(x)
        x = self.pixel_decoder_norm(x)

        # predictor projection
        x = self.pixel_decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_feature_decoder(self, x, ids_restore):
        # embed tokens
        x = self.feature_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.feature_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.feature_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.feature_decoder_blocks:
            x = blk(x)
        x = self.feature_decoder_norm(x)

        # predictor projection
        x = self.feature_decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        feature_decoder_pos_embed = get_2d_sincos_pos_embed(self.feature_decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.feature_decoder_pos_embed.data.copy_(torch.from_numpy(feature_decoder_pos_embed).float().unsqueeze(0))

        if self.args.use_pixel:
            pixel_decoder_pos_embed = get_2d_sincos_pos_embed(self.pixel_decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pixel_decoder_pos_embed.data.copy_(torch.from_numpy(pixel_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.pixel_mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        torch.nn.init.normal_(self.feature_mask_token, std=.02)


        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self,):
        print('error!')
    



def featmae_vit_tiny_patch4_dec5128b(**kwargs):
    model = FeatureMaskedAutoencoderViT(
        img_size=32,
        patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

