class MixCircleNetConfig():
    r"""
    Args:
        in_channels (`int`, *optional*, defaults to `3`):
            Number of input channels.
        out_channels (`int`, *optional*, defaults to `1`):
            Number of output channels.
        num_classes (`int`, *optional*, defaults to `2`):
            Number of classes (including background) for semantic segmentation.
        depth (`int`, *optional*, defaults to `5`):
            Depth of the U-Net architecture.
        start_filters (`int`, *optional*, defaults to `64`):
            Number of filters in the first convolutional layer.
        batch_norm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization after convolutional layers.
        dropout (`float`, *optional*, defaults to `0.0`):
            Dropout rate after each convolutional layer.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        n_classes=1,
        depth=5,
        n_filters=64,
        batch_norm=True,
        dropout=0,
        **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.depth = depth
        self.n_filters = n_filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        for k, v in kwargs.items():
            setattr(self, k, v)
