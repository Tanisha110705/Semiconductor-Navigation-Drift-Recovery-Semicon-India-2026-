##Run this script once locally to generate your model.onnx weight file:
import torch
import torch.nn as nn

class SEMFeatureExtractor(nn.Module):
    def __init__(self):
        super(SEMFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.encoder(x)

def main():
    model = SEMFeatureExtractor()
    model.eval()
    dummy_input = torch.randn(1, 1, 1000, 1000)
    
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    print("✅ Model weights exported to 'model.onnx'")

if __name__ == "__main__":
    main()
