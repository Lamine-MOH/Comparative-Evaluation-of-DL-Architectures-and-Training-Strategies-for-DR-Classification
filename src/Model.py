import torch
import torch.nn as nn
from torchvision import models as tv_models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def model_setup(name, num_classes, conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if name == "InceptionV3Scratch":
        model = InceptionV3Scratch(num_classes).to(device, non_blocking=True)

    # Loss Function
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Optimizer
    optimizer = Adam(model.parameters(), lr=conf['LR'])
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', factor=conf['LR_SCHED_FACTOR'],
        patience=conf['LR_SCHED_PAT'], min_lr=conf['LR_MIN']
    )
     
    # Return
    return {
        'device': device,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler
    }


def count_params(m):
    return sum(p.numel() for p in m.parameters())

class InceptionV3Scratch(nn.Module):
    """
    """
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone (aux_logits disabled)
        backbone = tv_models.inception_v3(init_weights=False,
                                          weights=None,
                                          aux_logits=True)

        # Store backbone without its original FC
        in_features = backbone.fc.in_features 
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.backbone(x)
        
        if isinstance(x, tv_models.inception.InceptionOutputs):
            x = x.logits  # Extract the primary output tensor
            
        x = self.dropout(x)
        return self.fc(x)
