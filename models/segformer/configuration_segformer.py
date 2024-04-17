class SegformerConfig():
    def __init__(self, 
        in_channels=3,
        widths=[64, 128, 256],
        depths=[3, 4, 6],
        all_num_heads=[4,4,4],
        patch_sizes=[3,5,3],
        overlap_sizes=[2, 2, 2],
        reduction_ratios=[8, 4, 2],
        mlp_expansions=[4,4,4],
        decoder_channels=256,
        scale_factors=[8, 4, 2],
        num_classes=1,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.widths = widths
        self.depths = depths
        self.all_num_heads = all_num_heads
        self.patch_sizes = patch_sizes
        self.overlap_sizes = overlap_sizes
        self.reduction_ratios = reduction_ratios
        self.mlp_expansions = mlp_expansions
        self.decoder_channels = decoder_channels
        self.scale_factors = scale_factors
        self.num_classes = num_classes
        

