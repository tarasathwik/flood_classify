from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define CNNModel again here (same as training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
    	x = self.pool(torch.relu(self.conv1(x)))
    	x = self.pool(torch.relu(self.conv2(x)))
    	print("Shape before flattening:", x.shape)
    	x = x.view(x.size(0), -1)
    	print("Shape after flattening:", x.shape)
    	x = torch.relu(self.fc1(x))
    	x = self.fc2(x)
    	return x



# Load the model
model = CNNModel()
model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Flask app
app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        result = 'Flooded' if predicted.item() == 0 else 'Non-Flooded'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
