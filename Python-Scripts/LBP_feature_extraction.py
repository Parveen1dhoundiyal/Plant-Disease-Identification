import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from PIL import Image
from tqdm import tqdm

# Set the path to  image dataset
dataset_path = "/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/PlantDiseasesDataset"
labels = []
features = []

# LBP parameters
radius = 3
n_points =24
method = 'uniform'

# Loop through each image in the dataset
for folder_name in os.listdir(dataset_path):
    label = folder_name
    folder_path = os.path.join(dataset_path, folder_name)

    for image_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
        image_path = os.path.join(folder_path, image_name)

        # Load the image using PIL to ensure RGBA images are properly handled
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image (resize to a common size)
        resized_image = image.resize((100, 100))

        # Convert the resized image to grayscale
        gray_image = resized_image.convert("L")

        # Calculate LBP texture feature
        lbp_image = local_binary_pattern(np.array(gray_image), n_points, radius, method)
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= lbp_hist.sum()   # Normalize

        features.append(lbp_hist)
        labels.append(label)

# Create a DataFrame to store the features and labels
data = pd.DataFrame(features)
data['label'] = labels
# Shuffle the DataFrame
data = data.sample(frac=1).reset_index(drop=True)
# Save the DataFrame to a CSV file
data.to_csv('/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Python-Scripts/model_0_features.csv', index=False)

