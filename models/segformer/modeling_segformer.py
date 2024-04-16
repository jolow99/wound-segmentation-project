import torch.nn as nn 
import torch 
import einops 

class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(num_features)
    def forward(self, x): 
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.ln(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x
    
class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size):
        super(OverlapPatchMerging, self).__init__()
        self.overlapPatchMergeConv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=patch_size, 
            stride=overlap_size, 
            padding=patch_size//2,
            bias=False
        ) 
        self.layerNorm = LayerNorm2d(out_channels)
    def forward(self, input): 
        x = self.overlapPatchMergeConv(input)
        x = self.layerNorm(x)
        return x 
    
class EfficientMultiHeadAttention(nn.Module): 
    def __init__(self, channels, reduction_ratio, num_heads): 
        super(EfficientMultiHeadAttention, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio),
            LayerNorm2d(channels)
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.size()
        q = einops.rearrange(x, 'b c h w -> b (h w) c')
        k = v = einops.rearrange(self.reducer(x), 'b c h w -> b (h w) c')
        out = self.att(q, k, v)[0]
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out 

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class MixFFN(nn.Module): 
    def __init__(self, channels, expansion_factor=4): 
        super(MixFFN, self).__init__()
        self.mlp = nn.Conv2d(channels, channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(channels, channels*expansion_factor, kernel_size=3, padding=1, groups=channels)
        self.gelu = nn.GELU()
        self.expansion_mlp = nn.Conv2d(channels*expansion_factor, channels, kernel_size=1)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.depthwise_conv(x)
        x = self.gelu(x) 
        out = self.expansion_mlp(x)
        return out

class ResidualSkipConnection(nn.Module): 
    def __init__(self, sublayer): 
        super(ResidualSkipConnection, self).__init__()
        self.sublayer = sublayer
    def forward(self, x, **kwargs): 
        return x + self.sublayer(x, **kwargs) 

class SegFormerEncoderBlock(nn.Module):
    def __init__(self, 
        channels, 
        reduction_ratio=1, 
        num_heads=8,
        expansion_factor=4
    ): 
        super(SegFormerEncoderBlock, self).__init__()
        self.mlp = MixFFN(channels, expansion_factor)
        self.att = EfficientMultiHeadAttention(channels, reduction_ratio, num_heads)

        self.encoder_block = nn.Sequential(
            ResidualSkipConnection(
                nn.Sequential(
                    LayerNorm2d(channels),
                    self.att
                )
            ),
            ResidualSkipConnection(
                nn.Sequential(
                    LayerNorm2d(channels),
                    self.mlp
                )
            )
        ) 

    def forward(self, x):
        return self.encoder_block(x)
    
class SegFormerEncoderLayer(nn.Sequential): 
    def __init__(
        self, 
        in_channels,
        out_channels,
        patch_size,
        overlap_size,
        depth=2,
        reduction_ratio=1,
        num_heads=8,
        mlp_expansion=4
    ):
        super(SegFormerEncoderLayer, self).__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)
    def forward(self, x):
        x = self.overlap_patch_merge(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        widths,
        depths,
        all_num_heads,
        patch_sizes,
        overlap_sizes,
        reduction_ratios,
        mlp_expansions,
    ):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderLayer(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features    
    
class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels, widths, scale_factors):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        new_features = []
        for feature, stage in zip(features,self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features
    
class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels) 
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x
    
class SegFormer(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            config.in_channels,
            config.widths,
            config.depths,
            config.all_num_heads,
            config.patch_sizes,
            config.overlap_sizes,
            config.reduction_ratios,
            config.mlp_expansions,
        )
        self.decoder = SegFormerDecoder(config.decoder_channels, config.widths[::-1], config.scale_factors)
        self.head = SegFormerSegmentationHead(
            config.decoder_channels, config.num_classes, num_features=len(config.widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation
