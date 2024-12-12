# Facial Skin Analysis and Diagnosis Using Inception V3

This project utilizes deep learning techniques to analyze and diagnose facial skin diseases using the Inception V3 model. The model classifies skin conditions based on images and provides diagnosis predictions with high accuracy.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

Facial skin conditions such as acne, eczema, rosacea, and others are common and can often be diagnosed with the help of machine learning models. In this project, we developed a skin disease classification system using the Inception V3 architecture, which achieves high accuracy in identifying skin conditions from facial images.

## Features

- Classifies skin diseases from facial images
- 92% accuracy across seven different classes
- Real-time predictions for skin condition diagnosis
- User-friendly interface for uploading images and receiving results

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/skin-disease-diagnosis.git
   cd skin-disease-diagnosis
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The model is trained on a custom dataset containing facial skin disease images. The dataset is divided into the following categories:

- Acne
- Eczema
- Psoriasis
- Rosacea
- Melasma
- Vitiligo
- Wrinkles

The dataset can be downloaded from [link to dataset].

## Model Architecture

The Inception V3 architecture was chosen due to its efficiency in extracting features from images and its success in various image classification tasks. The model was fine-tuned using transfer learning, with pre-trained weights on ImageNet.

Key components:
- Pre-trained Inception V3 model
- Fine-tuning on the skin disease dataset
- Custom output layer for classification

## Usage

1. Run the model inference by uploading an image of a face:

   ```bash
   python predict.py --image_path /path/to/image.jpg
   ```

2. The script will output the predicted skin condition along with the associated probability.

## Results

The model achieved an accuracy of 92% across the seven classes. Below is an example of the predicted output:

```text
Predicted Class: Acne
Probability: 89.5%
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
