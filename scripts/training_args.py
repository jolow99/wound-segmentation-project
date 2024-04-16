import torch 

class TrainingArguments:
    def __init__(
        self, 
        output_dir='output', 
        num_train_epochs=3, 
        warmup_steps=500, 
        weight_decay=0.01, 
        learning_rate=5e-5,
        lr_scheduler_type='linear',
        logging_dir='logs', 
        logging_steps=100, 
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        optim=torch.optim.AdamW,
        **kwargs
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.optim = optim
        self.weight_decay = weight_decay
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
