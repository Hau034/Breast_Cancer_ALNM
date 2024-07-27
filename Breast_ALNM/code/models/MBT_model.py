"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


'''
把二维图像转换成patch的形式 得到 [B,196,768]
'''


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    '''
    参数1：img_size,图像大小
    参数2：patch_size，2D图像的每一个patch长宽
    参数3：in_c，输入图像的维度
    参数4：embed_dim，经过卷积之后输入特征层的个数
    参数5：norm_layer，层标准化，和CNN的BN有所不同
    '''

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 初始化一些参数
        img_size = (img_size, img_size)  # 图像的大小
        patch_size = (patch_size, patch_size)  # 每一个patch 的长款
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 14*14
        # 一个patch 展开一维之后向量的长度
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 卷积，其中这个proj
        '''
        卷积，当输入in_channel = in_c=3,输出 out_channel = embed_dim = 768, 卷积核大小是 16x16,stride 大小是 16x16
             那么卷积的时候滑动刚好滑动一个窗口计算一个patch的大小。
             计算 224x224 的图像经过此卷积之后图像的大小。
             H_out ={[ H_in + 2 x pandding[0] - dilation[0] x (kernel_size[0] - 1) - 1 ]/ stride[0]} +1  其中 dilation默认为1
             W_out ={[ H_in + 2 x pandding[1] - dilation[1] x (kernel_size[1] - 1) - 1 ]/ stride[1]} +1
             故 
             
             H_out = W_out = [(224 + 2*0 - 1 * (16-1) -1 ] / 16 + 1 = （224 - 16）/16 + 1 = 13+1 = 14
        
        '''
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 在增减网络的过程中我们就可以用identity占个位置，这样网络整体层数永远不变， norm_layer 默认为None
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 首先输入数据，x
        # print(type(x))
        # 这一句会出现一个警告 TracerWarning: Converting a tensor to a NumPy array might cause the trace to be incorrect
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # proj: [B,C,H,W] = [B,3,224,224] -> [B,C,H,W] = [B,768,14,14] #
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C] = [B,196,768] 对输入的图像进行映射，映射到一个到维度，映射的参数也是可训练的
        '''
        flatten(2) 代表从第二个维度开始展平，也就是 从H开始、
        使用 transpose 对维度1,2上的数据进行调换
        最终变换成维度： [num_token,token_dim]=[B,196,768] ,其中每一个num_token 对应着一个图片的patch
        '''
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


'''
经过attention 机制 得到相同输入的大小 [B,197,768]
'''


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # mutil_head 的头的个数
        head_dim = dim // num_heads  # 计算真实就是那个过程中head的个数（每一个head对应的dimention)
        self.scale = qk_scale or head_dim ** -0.5  # qk_scale不指定，使用  head_dim 作为比例
        '''
        qkv的计算，无论是否有分 mutil_head 这个都是一起计算的，后面才把不同的head分出来
        一个输入得到三个输出，且矩阵 Wq,Wk,Wv 都是可以训练的参数，故这里也是使用线性层进行代替。
        
        '''
        self.qkv = nn.Linear(dim, dim * 3,
                             bias=qkv_bias)  # 并行计算qkv,输入就是 dim(768 维度相当于x1---x767)，进过 Wq,Wk,Qv的映射之后，得到 q,k,v

        self.attn_drop = nn.Dropout(attn_drop_ratio)  # dropout层
        self.proj = nn.Linear(dim, dim)  # 线性层
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # dropout层

    # 一对一的输出
    def forward(self, x):
        '''
        在这一步之前一张图像
        1.首先经过了 Embedding 层得到了图像的 [batch_size,num_token,token_dim]= [196,768]
        2.加上类别信息（contact）: [batch_size,num_token,token_dim] -> [batch_size,num_token+1,token_dim]
        3.加上位置信息 (add): [batch_size,num_token+1,token_dim]
        '''

        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv():  [batch_size, num_patches + 1, total_embed_dim] -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head] 目的是把 Q,K，V 分出来
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        '''
        在经过了前面把输入映射到了高纬度之后得到了 [batch_size,num_token+1,token_dim]
        1. 计算得到 QKV,这个是一次性计算，得到的 [batch_size, num_patches + 1, total_embed_dim]=[btch_size,197,768] -> [batch_size, num_patches + 1, 3 * total_embed_dim]=[btch_size,197,768*3] 
        2. 对 得到qkv 进行reshape [btch_size,197,768*3] -> [batch_size,197,3,num_heads,768/num_heads]
        3. permute [batch_size,197,3,num_heads,768/num_heads] ->[3,batch_size,num_heads,197,768/num_heads]
        4. 根据 num_head 的个数，把对qkv进行均分(但是注意，Batch_szie和 patch始终是绑定在一起的），对最后768*3这个维度进行均分 先把 3这个维度放在前面，代表的是QKV这三个，后面这两个维度是QKV (per_head,per_head_num) 
        5. 把 3这个维度放在最前面，拿出第一个维度就是 q,第二个维度就是 k,第三个维度即使v
        
        '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head] # 把每一个头分出来num_heads 放在最前面
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        '''
        下面对每一个 head 分开计算
        '''

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 这一步是 q乘于k的装置，除于 根号 k 的维度
        attn = attn.softmax(dim=-1)  # 进行softmax激活
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # softmax的结果乘于 v ,经过 self-attention之后还是的到输入一样的维度
        x = self.proj(x)  # 经过线性层，也就是 乘上Wo 对数据记性分离 B, N, C
        x = self.proj_drop(x)  # doropout层
        return x


'''
线性层 输出还是 [B,197,768]
'''


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果没有传入outfeature 那么默认等于 in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性层
        self.act = act_layer()  # 激活层
        self.fc2 = nn.Linear(hidden_features, out_features)  # 线性层
        self.drop = nn.Dropout(drop)  # dropout层

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''
Transformer 的Encoder Block, 这个是VisionTransformer 的关键

这里的流程是跟笔记对应的

'''


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,  # 第一个全连接层的输出是输入的4倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        '''
        1. 对输入经过 layer_norm 层，其中 dims是输入的768
        2. 经过self-Attention机制，需要输入两个参数，输入的dim代销，head的大小，自动计算qkv,对[batch_size,num_token,token_dim] 进行self-attention 计算之后返回的，其维度不变 [batch_size,num_token,token_dim]
        3. 经过 dropPath
        4. 和原始输入进行 skip_connect
        5. 经过layer_norm 
        6. 经过 MLP_Block 之后输出特征通道增加4倍
        7. 再次经过 dropPath
        8.和 4步骤的输出进行相加
        '''
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()  # 如果drop_path_ratio大于0 进行dropPath，否则使用 占位符进行占位
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP 第一个全连接层的输出

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x









class MBT_2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, fusion_layer=7, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            fusion_layer: Which layer to fuse modalities.
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(MBT_2, self).__init__()

        # 融合的层
        self.fusion_layer = fusion_layer
        # 这个只有一份
        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 判断融合的层和层数的关系,融合的层不能大于深度，小于0
        assert self.fusion_layer <= depth and self.fusion_layer >= 0

        self.patch_embed_bmode = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.patch_embed_swe = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)

        num_patches = self.patch_embed_bmode.num_patches

        self.count = 0
        self.scale = nn.Parameter(torch.FloatTensor([0.5]))

        self.cls_token_bmode = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_swe = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 这个只有一份
        self.dist_token_bmode = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.dist_token_swe = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed_bmode = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_swe = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop_bmode = nn.Dropout(p=drop_ratio)
        self.pos_drop_swe = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks_bmode = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.blocks_swe = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm_bmode = norm_layer(embed_dim)
        self.norm_swe = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits_bmode = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
            self.pre_logits_swe = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits_bmode = nn.Identity()
            self.pre_logits_swe = nn.Identity()

        # Classifier head(s)
        self.head_2 = nn.Linear(self.num_features * 2, num_classes) if num_classes > 0 else nn.Identity()
        self.head_1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.out2 = nn.Linear(self.num_features + 500, self.num_classes) if num_classes > 0 else nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed_bmode, std=0.02)
        if self.dist_token_bmode is not None:
            nn.init.trunc_normal_(self.dist_token_bmode, std=0.02)

        nn.init.trunc_normal_(self.cls_token_bmode, std=0.02)
        self.apply(_init_vit_weights)
        self.dropout = nn.Dropout(p=0.1)

    def forward_features(self, x1, x2):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x1 = self.patch_embed_bmode(x1)  # [B, 196, 768]
        x2 = self.patch_embed_swe(x2)  # [B, 196, 768]

        # [1, 1, 768] -> [B, 1, 768]
        cls_token_bmode = self.cls_token_bmode.expand(x1.shape[0], -1, -1)
        cls_token_swe = self.cls_token_swe.expand(x1.shape[0], -1, -1)

        if self.dist_token_bmode is None:
            x1 = torch.cat((cls_token_bmode, x1), dim=1)  # [B, 197, 768] # 在第0个维度加入类别信息
            x2 = torch.cat((cls_token_swe, x2), dim=1)  # [B, 197, 768] # 在第0个维度加入类别信息
        else:
            x1 = torch.cat((cls_token_bmode, self.dist_token_bmode.expand(x1.shape[0], -1, -1), x1), dim=1)
            x2 = torch.cat((cls_token_swe, self.dist_token_swe.expand(x2.shape[0], -1, -1), x1), dim=1)

        x1 = self.pos_drop_bmode(x1 + self.pos_embed_bmode)
        x2 = self.pos_drop_swe(x2 + self.pos_embed_swe)

        my_param_limited = torch.sigmoid(self.scale)

        for i in range(12):
            x1 = self.blocks_bmode[i](x1)
            x2 = self.blocks_swe[i](x2)
            if i >= self.fusion_layer:
                temp1 = x1
                temp2 = x2
                x1 = temp1 + my_param_limited * temp2
                x1 = self.norm_bmode(x1)  # norm_layer
                x2 = temp2 + my_param_limited * temp1
                x2 = self.norm_swe(x2)  # norm_layer


        if self.dist_token_bmode is None:
            # 去除第一个batch 维度，第二个维度索引为 0的数据，也就是前面加入的类别信息
            return self.pre_logits_bmode(x1[:, 0]), self.pre_logits_swe(x2[:, 0])
        else:
            return x1[:, 0], x1[:, 1]

    def forward(self, x1, x2):

        x1, x2 = self.forward_features(x1, x2)
        # my_param_limited = torch.sigmoid(self.scale)

        # x3 = torch.add(x1, x2)
        x3 = torch.add(x1, x2)
        x = self.head_1(x3)

        return x




def MBT_in21k_bmodeswe(num_classes: int = 21843, fusion_layer: int = 7, has_logits: bool = True):

    model = MBT_2(img_size=224,
                  patch_size=16,
                  embed_dim=768,
                  fusion_layer=fusion_layer,
                  depth=12,
                  num_heads=12,
                  representation_size=768 if has_logits else None,
                  num_classes=num_classes)
    return model



def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def reconstruct_model_parameters(weight_path: str, inPrintPthKey: bool, model, isPrintModelKey: bool, save_path: str):
    # 创建一个新的字典用于保存新的 model 的参数
    new_dict = {}
    '''
    include_blocks,nor_include_blocks 分别获取参数层带 blocks 名字和不带blocks 层的列表blocks_index，nor_blocks_index 分别代表他们的索引
    '''
    include_blocks = []
    nor_include_blocks = []
    blocks_index = 0
    nor_blocks_index = 0
    # 加载模型参数
    weights_dict = torch.load(weight_path, map_location=device)
    # 打印模型文件的参数
    BMT_model_name = []

    if isPrintModelKey:
        # 打印模型的名称和输出的形状
        for name, parameters in model.named_parameters():
            print(name, ';', parameters.size())
            BMT_model_name.append(name)

    if inPrintPthKey:
        for key in list(weights_dict.keys()):
            print(key, ";", weights_dict[key].shape)
            print(key, ";", weights_dict[key].shape)
    # 复制权重文件的数值到 new_dict 里面
    for index, name in enumerate(weights_dict.keys()):
        if "blocks" in name:
            blocks_index += 1
            rename_bmode = name.replace("blocks", "blocks_bmode")
            rename_swe = name.replace("blocks", "blocks_swe")
            print(f"index:{blocks_index}/{len(weights_dict.keys())},name:{name}")
            # 增加字典，删除字典
            new_dict[rename_swe] = new_dict[rename_bmode] = weights_dict[name]
            include_blocks.append(name)
        else:
            nor_blocks_index += 1
            print(f"index:{nor_blocks_index}/{len(weights_dict.keys())},name:{name}")
            if "cls_token" == name:
                new_dict["cls_token_swe"] = new_dict["cls_token_bmode"] = weights_dict[name]
            elif "norm" in name:
                rename_bmode = name.replace("norm", "norm_bmode")
                rename_swe = name.replace("norm", "norm_swe")
                new_dict[rename_swe] = new_dict[rename_bmode] = weights_dict[name]
            elif "patch_embed" in name:
                rename_bmode = name.replace("patch_embed", "patch_embed_bmode")
                rename_swe = name.replace("patch_embed", "patch_embed_swe")
                new_dict[rename_swe] = new_dict[rename_bmode] = weights_dict[name]
            elif "pos_embed" in name:
                rename_bmode = name.replace("pos_embed", "pos_embed_bmode")
                rename_swe = name.replace("pos_embed", "pos_embed_swe")
                new_dict[rename_swe] = new_dict[rename_bmode] = weights_dict[name]

            nor_include_blocks.append(name)
    # 保存新的权重文件
    torch.save(new_dict, save_path)

    return include_blocks, nor_include_blocks

from src_utils.utils import get_path
Brease_Cancer_pytorch_path_upDir = get_path(2)
print(Brease_Cancer_pytorch_path_upDir)
if __name__ == '__main__':

    #什么意思
    create_model = MBT_in21k_bmodeswe_2  # in21k 数据集中是包含 has_logits 层的


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2, fusion_layer=7, has_logits=False).to(device)

    # 打印模型的名字和尺寸 E:\SynologyDrive_workStation\1_预训练权重\ViT\单模态
    include_blocks, nor_include_blocks = reconstruct_model_parameters(
        weight_path=Brease_Cancer_pytorch_path_upDir + "/1_预训练权重/ViT/单模态/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
        inPrintPthKey=False,
        model=model,
        isPrintModelKey=False,
        save_path="./new_model.pth")

    print("=========================")
    print(f"权重层包含 blocks 名字的个数：{len(include_blocks)} ")
    print(f"权重层不包含 blocks 名字的个数：{len(nor_include_blocks)} ")
    for data in nor_include_blocks:
        print(data)
    print("=========================")

    #加载原始模型参数
    ori_dict = torch.load(Brease_Cancer_pytorch_path_upDir + "/1_预训练权重/ViT/单模态/jx_vit_base_patch16_224_in21k-e5005f0a.pth")

    changed_dict = torch.load('./new_model.pth')

    print(f"原始权重层的个数：{len(ori_dict.keys())}")

    for key in list(ori_dict.keys()):
        print(key, ";", ori_dict[key].shape)

    print(f"重构之后权重层的个数：{len(changed_dict.keys())}")

    for key in list(changed_dict.keys()):
        print(key, ";", changed_dict[key].shape)
