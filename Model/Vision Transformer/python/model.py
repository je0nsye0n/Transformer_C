import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from einops import rearrange
from collections import OrderedDict

###############################################################################
# DropPath (Stochastic Depth)
###############################################################################
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

###############################################################################
# MLP (Feed Forward) as a Sequential with keys "0", "1", "2", "3"
###############################################################################
class MLP(nn.Sequential):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        layers = OrderedDict([
            ("0", nn.Linear(in_features, hidden_features)),
            ("1", nn.GELU()),
            ("2", nn.Dropout(dropout)),
            ("3", nn.Linear(hidden_features, in_features))
        ])
        super().__init__(layers)

###############################################################################
# Encoder Layer (torchvision과 동일한 구조 및 키 이름)
###############################################################################
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float,
                 dropout: float, attn_dropout: float, drop_path: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=attn_dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attention(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

###############################################################################
# Encoder 모듈: positional embedding, encoder layers, 최종 norm 포함
###############################################################################
class Encoder(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, depth: int,
                 num_heads: int, mlp_ratio: float, dropout: float,
                 attn_dropout: float, drop_path_rate: float):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        layers = OrderedDict()
        for i in range(depth):
            layers[f"encoder_layer_{i}"] = EncoderLayer(embed_dim, num_heads, mlp_ratio,
                                                          dropout, attn_dropout, dpr[i])
        self.layers = nn.ModuleDict(layers)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)
        x = self.ln(x)
        return x

###############################################################################
# VisionTransformer 구현 (키 구조를 torchvision vit_b_16과 동일하게)
###############################################################################
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        # Patch embedding (이름: conv_proj)
        self.conv_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Encoder: pos_embedding, layers, ln
        self.encoder = Encoder(num_patches, embed_dim, depth, num_heads, mlp_ratio, dropout, attn_dropout, drop_path_rate)
        self.pos_drop = nn.Dropout(dropout)
        # Classification head (이름: heads.head)
        self.heads = nn.Sequential(OrderedDict([("head", nn.Linear(embed_dim, num_classes))]))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.encoder.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embedding
        x = self.conv_proj(x)                   # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, embed_dim)
        # Prepend class token
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)     # (B, num_patches+1, embed_dim)
        # Add positional embedding
        x = x + self.encoder.pos_embedding
        x = self.pos_drop(x)
        # Encoder layers
        x = self.encoder(x)
        # Classification head uses class token (index 0)
        x = self.heads(x[:, 0])
        return x

###############################################################################
# 모델 생성 및 pretrained weight 적용
###############################################################################
weights = ViT_B_16_Weights.IMAGENET1K_V1
transform = weights.transforms()

custom_model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.0,
    attn_dropout=0.0,
    drop_path_rate=0.0
)

pretrained_model = vit_b_16(weights=weights)
pretrained_state_dict = pretrained_model.state_dict()

# state_dict 키가 완전히 일치하므로 바로 로드 가능
custom_model.load_state_dict(pretrained_state_dict)
custom_model.eval()
model = custom_model

###############################################################################
# 평가 코드 (ImageNet 데이터 기반)
###############################################################################
imagenet_labels = weights.meta["categories"]

data_dir = './Data'
image_extensions = ('.jpeg', '.jpg', '.png')
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]

log_path = './Model_Resultlog.txt'
with open(log_path, 'w', encoding='utf-8') as log_file:
    if not image_files:
        message = "데이터 폴더에 이미지가 없습니다."
        print(message)
        log_file.write(message + "\n")
    else:
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                x = transform(img).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x)
                    probabilities = F.softmax(logits, dim=-1)
                    pred_idx = probabilities.argmax(dim=-1).item()
                pred_label = imagenet_labels[pred_idx]
                result_str = f"이미지: {os.path.basename(img_path)} -> 예측 클래스: {pred_label} (인덱스: {pred_idx})"
                log_file.write(result_str + "\n")
            except Exception as e:
                error_msg = f"{os.path.basename(img_path)} 처리 중 오류 발생: {e}"
                print(error_msg)
                log_file.write(error_msg + "\n")
