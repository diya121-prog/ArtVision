import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.style_model import StyleModel
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Predict style of a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image.")
    parser.add_argument("--config", type=str, default="configs/train_style.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/style_model/best_model.pth")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.backends.mps.is_available() else "cpu")
    
    # Get classes from metadata
    df = pd.read_csv(config["data"]["metadata_path"])
    df = df[(df["split"] == "train") & (df["style"].notna())]
    classes = sorted(df["style"].dropna().unique().tolist())
    
    model = StyleModel(config["model"]["backbone"], config["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(args.image).convert("RGB")
    tensor = transform_eval(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze().numpy()
        
    top3_idx = probs.argsort()[-3:][::-1]
    
    print(f"\nPredictions for {args.image}:")
    for i, idx in enumerate(top3_idx):
        print(f"{i+1}. {classes[idx]} ({probs[idx]*100:.1f}%)")

if __name__ == "__main__":
    main()
