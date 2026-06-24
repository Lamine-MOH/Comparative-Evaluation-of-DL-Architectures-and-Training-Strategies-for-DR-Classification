import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tv_models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def model_setup(name, num_classes, conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ------ Build selected Model ------
    # Scratch vs Pretrained models
    if name == "InceptionV3Scratch":
        model = InceptionV3Scratch(num_classes)
    elif name == "InceptionV3Pretrained":
        model = InceptionV3Pretrained(num_classes)
    elif name == "ResNet50Scratch":
        model = ResNet50Scratch(num_classes)
    elif name == "ResNet50Pretrained":
        model = ResNet50Pretrained(num_classes)
    elif name == "VGG16Scratch":
        model = VGG16Scratch(num_classes)
    elif name == "VGG16Pretrained":
        model = VGG16Pretrained(num_classes)
    elif name == "AlexNetScratch":
        model = AlexNetScratch(num_classes)
    elif name == "AlexNetPretrained":
        model = AlexNetPretrained(num_classes)
    
    # State of the art models
    elif name == "ModifiedXceptionModel":
        model = ModifiedXceptionModel(num_classes)
    elif name == "HybridCNNSVDELM":
        model = HybridCNNSVDELM(num_classes)
    elif name == "MobileNetV2SVMClassifier":
        model = MobileNetV2SVMClassifier(num_classes)
    elif name == "EfficientNetB5DR":
        model = EfficientNetB5DR(num_classes)  
        
    # Move model to device
    model = model.to(device, non_blocking=True)

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

# ---------------------------------------------------------------------------
# Scratch vs Pretrained
# ---------------------------------------------------------------------------

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
    def __init__(self, num_classes=2, dropout=0.4, freeze_backbone=True):
        super().__init__(),
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
                                          aux_logits=True) # Changed from False to True

        # Store backbone without its original FC
        in_features = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()                    # remove classifier
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_partial(self, freeze_n_params):
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

    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes

        # Load untrained backbone
        backbone = tv_models.resnet50(weights=None) 

        # Store backbone without its original FC
        in_features = backbone.fc.in_features  
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)
    

class ResNet50Pretrained(nn.Module):

    def __init__(self, num_classes=2, dropout=0.4, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1) 

        # Store backbone without its original FC
        in_features = backbone.fc.in_features  
        backbone.fc = nn.Identity()      
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_from_layer(self, start_layer, freeze_bn=True):
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
    def __init__(self, num_classes=2, dropout=0.4, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load untrained backbone
        backbone = tv_models.vgg16_bn(weights=None) 

        # Store backbone without its original FC
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)
    

class VGG16Pretrained(nn.Module):
    """
    """
    def __init__(self, num_classes=2, dropout=0.4, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1) 

        # Store backbone without its original FC
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_from_block(self, start_block, freeze_bn=True):        
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
    def __init__(self, num_classes=2, dropout=0.4):
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
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
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
    def __init__(self, num_classes=2, dropout=0.4, freeze_backbone=True):
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
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
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
    

# ---------------------------------------------------------------------------
# State of Art
# ---------------------------------------------------------------------------

# -- ModifiedXceptionModel --------------------------------------------------

class ModifiedXceptionModel(nn.Module):
    """
    EfficientNet-B3 backbone with deep-layer aggregation and an MLP head.
 
    Args:
        num_classes:    Number of output classes (default 5 for APTOS grades).
        freeze_backbone: If True, backbone weights are frozen; only the
                         classifier head is trained.
    """
 
    # Channel widths at tapped EfficientNet-B3 feature blocks
    _TAP_CHANNELS = {2: 24, 5: 136, 7: 232}   # block index → out_channels
    _FINAL_CHANNELS = 1536                      # block 8 (final) out_channels
 
    def __init__(self, num_classes: int = 5, freeze_backbone: bool = False):
        super().__init__()
 
        backbone = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.DEFAULT
        )
        # Keep only the feature extractor blocks (drop the classifier head)
        self.features     = backbone.features          # nn.Sequential, blocks 0-8
        self.hook_indices = set(self._TAP_CHANNELS)   # {2, 5, 7}
 
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
 
        total_tapped = sum(self._TAP_CHANNELS.values()) + self._FINAL_CHANNELS  # 1928
 
        self.classifier = nn.Sequential(
            nn.Linear(total_tapped, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tapped = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.hook_indices:
                tapped.append(x)
        tapped.append(x)  # final feature map (block 8)
 
        # Global average pool each tapped map, then concatenate → (B, 1928)
        pooled     = [feat.mean(dim=[2, 3]) for feat in tapped]
        aggregated = torch.cat(pooled, dim=1)
 
        return self.classifier(aggregated)


# -- CNNFeatureExtractor ----------------------------------------------------

# CNN feature extractor  (Table 4 architecture)
class CNNFeatureExtractor(nn.Module):
    """
    Four conv-BN-pool-dropout blocks followed by two fully-connected layers.
 
    Args:
        in_channels:  Number of input channels (default 3 for RGB).
        n_features:   Dimension of the output feature vector (default 256).
    """
 
    def __init__(self, in_channels: int = 3, n_features: int = 256):
        super().__init__()
 
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
            )
 
        self.conv_blocks = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
 
        # After 4 × MaxPool2d(2,2) on a 224×224 input → 14×14 spatial size
        self.flatten = nn.Flatten()
 
        self.fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_features),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.flatten(x)
        return self.fc(x)
 
 
# ELM  (pure numpy — no gradient needed)
class ELM:
    """
    Extreme Learning Machine.
    
    Args:
        n_hidden:     Number of hidden neurons.
        activation:   'relu' | 'sigmoid' | 'tanh'.
        random_state: Seed for reproducibility.
    """
 
    def __init__(
        self,
        n_hidden: int = 500,
        activation: str = "relu",
        random_state: int = 42,
    ):
        self.n_hidden    = n_hidden
        self.activation  = activation
        rng              = np.random.default_rng(random_state)
        self._rng        = rng
        self.W    = None   # (n_features, n_hidden)  — set in fit()
        self.B    = None   # (1, n_hidden)
        self.beta = None   # (n_hidden, n_outputs)
 
    # ------------------------------------------------------------------
    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self.activation == "tanh":
            return np.tanh(x)
        return x  # linear
 
    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return self._activate(X @ self.W + self.B)   # (N, n_hidden)
 
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ELM":
        """
        Train ELM.
 
        Args:
            X: (N, n_features) float array.
            Y: (N, n_outputs)  float array  — one-hot for multiclass.
        """
        n_features = X.shape[1]
        self.W    = self._rng.standard_normal((n_features, self.n_hidden))
        self.B    = self._rng.standard_normal((1, self.n_hidden))
        H         = self._hidden(X)                  # (N, n_hidden)
        self.beta = np.linalg.pinv(H) @ Y            # (n_hidden, n_outputs)
        return self
 
    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw linear outputs  (N, n_outputs)."""
        return self._hidden(X) @ self.beta
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_scores(X)
        if scores.shape[1] == 1:
            return (scores > 0.5).astype(int).ravel()
        return np.argmax(scores, axis=1)
 
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_scores(X)
        if scores.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-scores))
            return np.hstack([1 - p, p])
        # Softmax
        shifted = scores - scores.max(axis=1, keepdims=True)
        e       = np.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)
 
 
# Full hybrid model
class HybridCNNSVDELM(nn.Module):
    """
    Two-phase hybrid model: CNN pre-training → SVD dimensionality reduction
    → ELM classification.

    Args:
        num_classes:   Number of target classes.
        in_channels:   Input image channels (default 3).
        n_cnn_features: CNN output feature dimension (default 256).
        n_svd_features: Reduced dimension after TruncatedSVD (default 100).
        n_elm_hidden:   ELM hidden neurons (default 500).
    """
 
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        n_cnn_features: int = 256,
        n_svd_features: int = 100,
        n_elm_hidden: int = 500,
    ):
        super().__init__()
 
        self.num_classes    = num_classes
        self.n_cnn_features = n_cnn_features
        self.n_svd_features = n_svd_features
 
        # --- CNN backbone ---
        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels,
            n_features=n_cnn_features,
        )
 
        # --- Phase-1 classification head (removed after SVD/ELM fitting) ---
        self._phase1_head = nn.Linear(n_cnn_features, num_classes)
 
        # --- SVD + ELM (fitted in phase 2, not part of the grad graph) ---
        self.scaler  = StandardScaler()
        self.svd     = TruncatedSVD(n_components=n_svd_features, random_state=42)
        self.elm     = ELM(n_hidden=n_elm_hidden)
        self._elm_fitted = False   # flag: which forward() path to use
 
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Phase 1: CNN logits  (used during train_model).
        Phase 2: ELM scores as a tensor  (used during evaluate_model).
        """
        features = self.cnn(x)   # (B, n_cnn_features)
 
        if not self._elm_fitted:
            # Phase 1 — differentiable path
            return self._phase1_head(features)
 
        # Phase 2 — non-differentiable ELM path; no grad needed
        with torch.no_grad():
            feats_np = features.cpu().numpy()
 
        feats_scaled = self.scaler.transform(feats_np)
        feats_svd    = self.svd.transform(feats_scaled)
        scores       = self.elm.decision_scores(feats_svd)   # (B, num_classes)
        return torch.from_numpy(scores.astype(np.float32)).to(x.device)
 
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_svd_elm(self, train_loader, device: torch.device) -> None:
        """
        Phase 2: extract CNN features for the whole training set, then
        fit StandardScaler → TruncatedSVD → ELM.
 
        Call this once after CNN pre-training is complete.
 
        Args:
            train_loader: DataLoader yielding (images, labels) batches.
            device:       Same device the model lives on.
        """
        print("Phase 2 — extracting CNN features for SVD/ELM fitting...")
        self.cnn.eval()
 
        all_features, all_labels = [], []
        for images, labels in train_loader:
            images = images.to(device)
            feats  = self.cnn(images).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())
 
        X = np.concatenate(all_features, axis=0)   # (N, n_cnn_features)
        y = np.concatenate(all_labels,   axis=0)   # (N,)
 
        # Standardise
        print(f"  Fitting StandardScaler + TruncatedSVD "
              f"({self.n_cnn_features} → {self.n_svd_features} features)...")
        X_scaled = self.scaler.fit_transform(X)
 
        # SVD
        self.svd.fit(X_scaled)
        cum_var = np.cumsum(self.svd.explained_variance_ratio_)[-1]
        print(f"  Cumulative explained variance: {cum_var:.4f}")
        X_svd = self.svd.transform(X_scaled)
 
        # One-hot labels for ELM
        Y_onehot = np.eye(self.num_classes)[y]     # (N, num_classes)
 
        # ELM
        print(f"  Fitting ELM ({self.elm.n_hidden} hidden neurons)...")
        self.elm.fit(X_svd, Y_onehot)
 
        self._elm_fitted = True
        print("  Phase 2 complete — model switched to ELM inference mode.")


# -- MobileNetV2 + SVM ----------------------------------------------------

# backbone + regression head 
class MobileNetV2Backbone(nn.Module):
    """
    Pre-trained MobileNetV2 with a two-layer FC head.

    Args:
        img_size:    Spatial resolution expected by the model (default 224).
        freeze_backbone: Freeze MobileNetV2 conv weights; train heads only.
    """

    def __init__(self, img_size: int = 224, freeze_backbone: bool = False):
        super().__init__()

        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Drop the original classifier;
        self.features = base.features          
        self.pool     = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Mirrors Keras
        self.fc1 = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 256),  nn.ReLU(inplace=True))

        # Regression output
        self.regression_head = nn.Linear(256, 1)

    # ------------------------------------------------------------------
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-d feature vector from fc2 (used by SVM in phase 2)."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x)
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.get_embedding(x))


# full pipeline model
class MobileNetV2SVMClassifier(nn.Module):
    """
    Args:
        num_classes:     Number of DR grades (default 5).
        img_size:        Input spatial resolution (default 224).
        freeze_backbone: Freeze MobileNetV2 conv weights during phase 1.
        svm_kernel:      SVM kernel — 'rbf' | 'linear' | 'poly'.
        svm_C:           SVM regularisation parameter.
        svm_gamma:       SVM gamma ('scale' | 'auto' | float).
    """

    def __init__(
        self,
        num_classes: int = 5,
        img_size: int = 224,
        freeze_backbone: bool = False,
        svm_kernel: str = "rbf",
        svm_C: float = 1.0,
        svm_gamma: str = "scale",
    ):
        super().__init__()

        self.num_classes = num_classes

        # CNN backbone with regression head
        self.backbone = MobileNetV2Backbone(
            img_size=img_size,
            freeze_backbone=freeze_backbone,
        )

        # SVM + scaler 
        self.scaler = StandardScaler()
        self.svm    = SVC(
            kernel=svm_kernel,
            C=svm_C,
            gamma=svm_gamma,
            probability=True,
            random_state=42,
            decision_function_shape="ovr",
        )
        self._svm_fitted = False

    def forward(self, x: torch.Tensor):
        if not self._svm_fitted:
            # standard regression forward
            return self.backbone(x)

        # SVM path
        with torch.no_grad():
            emb_np = self.backbone.get_embedding(x).cpu().numpy()

        emb_scaled = self.scaler.transform(emb_np)
        scores = self.svm.decision_function(emb_scaled).astype(np.float32)
        return torch.from_numpy(scores).to(x.device)

    @torch.no_grad()
    def fit_svm(self, train_loader, device: torch.device) -> None:
        """
        extract fc2 embeddings for the whole training set, fit
        StandardScaler, then fit SVM.

        Args:
            train_loader: DataLoader yielding (images, labels) batches.
                          Labels can be int or float — will be rounded and
                          cast to int for SVM fitting.
            device:       Device the model lives on.
        """
        self.backbone.eval()

        all_emb, all_labels = [], []
        for images, labels in train_loader:
            images = images.to(device)
            emb    = self.backbone.get_embedding(images).cpu().numpy()
            all_emb.append(emb)
            # Round float regression targets back to integer grades
            all_labels.append(
                np.round(labels.numpy()).astype(int).ravel()
            )

        X = np.concatenate(all_emb,    axis=0)   # (N, 256)
        y = np.concatenate(all_labels, axis=0)   # (N,)

        X_scaled = self.scaler.fit_transform(X)

        self.svm.fit(X_scaled, y)

        self._svm_fitted = True


# -- EfficientNet-B5 and CLAHE ----------------------------------------------------

class EfficientNetB5DR(nn.Module):
    """
    EfficientNet-B5 backbone with a two-layer classification head

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, freeze all backbone layers, only the
                          last `unfreeze_last_n` layers are kept trainable.
        unfreeze_last_n:  Number of backbone layers to keep trainable
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 20,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.binary      = (num_classes == 2)

        # backbone
        base = models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.DEFAULT
        )

        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1) 

        # freeze all backbone layers first
        for param in self.features.parameters():
            param.requires_grad = False

        # unfreeze the last
        if not freeze_backbone:
            all_layers = list(self.features.modules())
            for layer in all_layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # classification head
        backbone_out = 2048

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, num_classes) logits.

        For binary mode the two logits represent [non-referable, referable].
        BCEWithLogitsLoss in the wrapped criterion uses only logits[:, 1].
        Argmax over both logits is equivalent to thresholding at 0.5.
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)   # (B, 2048)
        return self.head(x)           # (B, num_classes)

# Loss wrapper for binary mode
class BinaryDRLoss(nn.Module):
    """
    Extracts the positive-class logit (index 1) before computing BCE
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        pos_logit = logits[:, 1]
        return self.bce(pos_logit, labels.float())
