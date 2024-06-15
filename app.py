import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# Define your model class (if needed) or use a predefined model
class ResNet34(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet34, self).__init__()
        self.network = models.resnet34(weights=None)  # No pre-trained weights
        num_ftrs = self.network.fc.in_features
        self.network.fc = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.network(x)

# Load your model
model = ResNet34()
model.load_state_dict(torch.load('skin-cancer-resnet34.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing steps
img_size = 224
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

# Define the labels for your classification
labels = [
  "Actinic keratosis",
 'Basal cell carcinoma',
 'Benign keratosis',
 'Dermatofibroma',
 'Melanocytic nevus',
 'Melanoma',
 'Squamous cell carcinoma',
 'Vascular lesion'
]

def predict(image):
    # Preprocess the image
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = labels[predicted.item()]

    return prediction

# Define the Gradio interface
inputs = gr.components.Image()
outputs = gr.components.Textbox(label="Prediction")

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Skin Cancer Detection",
             description="Upload an image of a skin lesion to get a prediction.").launch()
