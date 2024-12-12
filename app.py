from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from PIL import Image
import torch
from torchvision import models, transforms
import joblib  # To load the label encoder

app = Flask(__name__)

# Load your trained model
model = models.inception_v3(pretrained=False)
checkpoint = torch.load('skin_disease_model.pth')
model.load_state_dict(checkpoint, strict=False) 
model.eval()

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')  # Ensure you saved it as 'label_encoder.pkl'

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to process the image and make predictions
def process_image(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    # Process the image and make the prediction
    img_tensor = process_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = output.argmax(dim=1).item()
    
    # Convert the prediction back to the original class using the label encoder
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    
    # Define solutions for each disease (You can load this from a file or define it in the code)
    solutions = {
        'Acne': 'Use acne medication and a good skincare routine.',
        'Eksim': 'Apply moisturizing creams and avoid allergens.',
        'Herpes': 'Consult a doctor for antiviral treatments.',
        'Panu': 'Use antifungal creams and maintain hygiene.',
        'Rosacea': 'Avoid triggers and use prescription creams.'
    }

    solution = solutions.get(predicted_disease, "Consult a dermatologist for more information.")
    
    # Prepare the response data
    response_data = {
        'disease': predicted_disease,
        'solution': solution,
        'disease_image': f'/uploads/{file.filename}'  # Path to the uploaded image
    }
    
    return jsonify(response_data)

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
