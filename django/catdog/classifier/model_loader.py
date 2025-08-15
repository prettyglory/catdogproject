# classifier/model_loader.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Path to your trained model
MODEL_PATH = "C:/Users/USER/django/catdog/classifier/model/best_catdog_resnet18.pth"

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    """Load the ResNet-18 model with the correct fully connected layer."""
    model = models.resnet18(weights=None)  # no pretrained weights
    # Match the structure of your trained model
    model.fc = nn.Sequential(
        nn.ReLU(),                             # fc.0
        nn.Linear(model.fc.in_features, 2)     # fc.1 -> weights in your .pth
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # set to evaluation mode
    return model

# Load the model once when the module is imported
model = load_model()

def predict_image(image_data):
    """Predict whether the uploaded image is a cat or a dog."""
    # Open image
    image = Image.open(image_data).convert('RGB')
    # Preprocess
    img_tensor = preprocess(image).unsqueeze(0)  # add batch dimension
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return 'Dog' if predicted.item() == 1 else 'Cat'
