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

        # Load untrained backbone 
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


class InceptionV3Pretrained(nn.Module):
    """
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
                                          aux_logits=True) # Changed from False to True

        # Store backbone without its original FC
        in_features = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()                    # remove classifier
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_partial(self, freeze_n_params=FREEZE_LAYERS * 1000):
        # First unfreeze all
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Re-freeze leading parameters
        frozen = 0
        for param in self.backbone.parameters():
            if frozen < freeze_n_params:
                param.requires_grad = False
                frozen += param.numel()
            else:
                break
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f'  Backbone trainable params after partial unfreeze: {trainable:,}')

    def forward(self, x):
        x = self.backbone(x)

        if isinstance(x, tv_models.inception.InceptionOutputs):
            x = x.logits  # Extract the primary output tensor
            
        x = self.dropout(x)
        return self.fc(x)


class ResNet50Scratch(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # Load untrained backbone
        backbone = tv_models.resnet50(weights=None) 

        # Store backbone without its original FC
        in_features = backbone.fc.in_features  
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)
    

class ResNet50Pretrained(nn.Module):

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1) 

        # Store backbone without its original FC
        in_features = backbone.fc.in_features  
        backbone.fc = nn.Identity()      
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_from_layer(self, start_layer=FREEZE_START_LAYER, freeze_bn=True):
        # Define which named modules correspond to which "layer" numbers
        layer_names = {1: 'layer1', 2: 'layer2', 3: 'layer3', 4: 'layer4'}
        target_name = layer_names.get(start_layer, 'layer3')

        found_target = False
        for name, child in self.backbone.named_children():
            # If we hit the target layer (e.g., 'layer3'), we start unfreezing everything after
            if name == target_name:
                found_target = True
            
            if found_target:
                for param in child.parameters():
                    param.requires_grad = True
                
                # Special handling for BatchNorm within the unfrozen layers
                if freeze_bn:
                    for m in child.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval() # Use pretrained stats
                            for p in m.parameters():
                                p.requires_grad = False
            else:
                # Keep earlier layers frozen
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)


class VGG16Scratch(nn.Module):
    """
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load untrained backbone
        backbone = tv_models.vgg16_bn(weights=None) 

        # Store backbone without its original FC
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)
    

class VGG16Pretrained(nn.Module):
    """
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1) 

        # Store backbone without its original FC
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_from_block(self, start_block=FREEZE_START_BLOCK, freeze_bn=True):        
        block_mapping = {4: 17, 5: 24}
        limit = block_mapping.get(start_block, 17)

        for i, layer in enumerate(self.backbone.features):
            # Check if it's a Batch Norm layer
            if freeze_bn and isinstance(layer, nn.BatchNorm2d):
                layer.eval() # Force to use ImageNet stats
                for param in layer.parameters():
                    param.requires_grad = False
                continue # Skip to next layer to keep BN frozen globally
        
            # Handle Conv layers based on the block limit
            if i < limit:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)


class AlexNetScratch(nn.Module):
    """
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # Load untrained backbone
        backbone = tv_models.alexnet(weights=None) 

        # Store backbone without its original classifier
        # The input to the classifier after avgpool: 256 * 6 * 6 = 9216
        self.in_features = 9216 
        
        # Discard the original classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Custom head
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(self.in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if num_classes == 2 else num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetPretrained(nn.Module):
    """
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1) 

        # Store backbone without its original classifier
        # The input to the classifier after avgpool is 256 * 6 * 6 = 9216
        self.in_features = 9216 
        
        # Discard the original classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Custom head
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(self.in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if num_classes == 2 else num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
            
    def unfreeze_from_layer(self, start_conv_index=3):
        for i, layer in enumerate(self.features):
            if i >= start_conv_index:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x