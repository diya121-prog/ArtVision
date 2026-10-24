import sys
import torch
import torchvision
import timm

def main():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"Timm Version: {timm.__version__}")

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Dummy EfficientNet forward pass
    print("Running dummy EfficientNet forward pass...")
    try:
        model = timm.create_model("efficientnet_b0", pretrained=False)
        model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    main()
