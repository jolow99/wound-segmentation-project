class SegNetConfig():
    def __init__(
        self,
        n_filters=64, 
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
