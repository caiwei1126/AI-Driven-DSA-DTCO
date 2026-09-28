# model.py — CNN architecture (keys block1..block4 match best_model_4_CNN_fixed.pth)

# All conventions needed for reproduction:
#   1. Dataset: unzip split_dataset_fixed_data.zip into
#      {train,val,test}/{defect,perfect}/*.npy; each .npy is a (1,128,128)
#      float32 array with values in 0~1; map back to [-1,1] at load time via
#      x = (x - 0.5) * 2.
#   2. Labels: defect -> 0, perfect -> 1; score = sigmoid(logit) = P(perfect).
#   3. Decision threshold: 0.73.
import torch
import torch.nn as nn


class CNN4Layer(nn.Module):
    """4-layer CNN for defect/perfect binary classification.

    Input:  (B, 1, 128, 128) float32, range [-1, 1]
    Output: (B,) logits; P(perfect) = sigmoid(logit), P(defect) = 1 - that

    Architecture: conv(1->32) -> conv(32->64) -> maxpool -> dropout2d
                  -> conv(64->96) -> conv(96->512) -> GAP -> linear(512->1)
    Each conv is followed by BatchNorm2d + LeakyReLU(0.1), k=3, s=1, p=1.
    Always call model.eval() for inference (BatchNorm/Dropout inference mode).
    """

    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(negative_slope=0.1),
            )
        self.block1 = block(1, 32)
        self.block2 = block(32, 64)
        self.pool   = nn.MaxPool2d(2)
        self.drop   = nn.Dropout2d(0.25)
        self.block3 = block(64, 96)
        self.block4 = block(96, 512)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, 1)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.drop(self.pool(x))
        x = self.block3(x); x = self.block4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x).squeeze(1)


def load_model(weights_path, device="cpu"):
    """Load the best CNN weights; device 'cpu' or 'cuda:0' gives identical results."""
    model = CNN4Layer()
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()
