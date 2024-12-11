import ast
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from torchvision.models import resnet18, alexnet, vgg16
from torchvision.models import ResNet18_Weights, AlexNet_Weights, VGG16_Weights


models = {
    'resnet': resnet18(weights=ResNet18_Weights.DEFAULT),
    'alexnet': alexnet(weights=AlexNet_Weights.DEFAULT),
    'vgg': vgg16(weights=VGG16_Weights.DEFAULT)
}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def classifier(img_path, model_name):
    if model_name not in models:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Get the pre-initialized model
    model = models[model_name]

    # Load the image
    img_pil = Image.open(img_path).convert("RGB")

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess the image
    img_tensor = preprocess(img_pil).unsqueeze(0)

    # Move model and tensor to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    img_tensor = img_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)

    # Get top-1 prediction
    pred_idx = output.argmax(dim=1).item()
    return imagenet_classes_dict.get(pred_idx, "Unknown")
