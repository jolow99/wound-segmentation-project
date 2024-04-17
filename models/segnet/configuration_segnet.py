class UNetConfig():
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
        n_filters=32, 
        input_dim_x=224, 
        input_dim_y=224, 
        num_channels=3,
        **kwargs
    ):
        self.n_filters = n_filters
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.num_channels = num_channels
        for k, v in kwargs.items():
            setattr(self, k, v)
