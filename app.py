import os
import streamlit as st
from fastai.learner import load_learner
from PIL import Image
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

# Define the folder path and filename
folder_path = 'models'
fname = 'sea_turtle_model.pkl'

# Define the model path
model_path = os.path.join(folder_path, fname)

# Define a function to load the trained model
def load_model(model_path):
    learn = load_learner(model_path, cpu=True)
    return learn

# Define the IOU function
def iou(boxA, boxB):
    # Extract coordinates from boxA
    xA1, yA1, wA, hA = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]
    xA2, yA2 = xA1 + wA, yA1 + hA
    
    # Extract coordinates from boxB
    xB1, yB1, wB, hB = boxB[:, 0], boxB[:, 1], boxB[:, 2], boxB[:, 3]
    xB2, yB2 = xB1 + wB, yB1 + hB
    
    # Determine the intersection rectangle
    xA = torch.max(xA1, xB1)
    yA = torch.max(yA1, yB1)
    xB = torch.min(xA2, xB2)
    yB = torch.min(yA2, yB2)

    # Calculate the intersection area
    interArea = torch.clamp(xB - xA + 1, min=0) * torch.clamp(yB - yA + 1, min=0)

    # Calculate the areas of both bounding boxes
    boxAArea = (wA + 1) * (hA + 1)
    boxBArea = (wB + 1) * (hB + 1)

    # Calculate the IOU
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou.mean()  # Average IOU over all items in batch

# Define a function to make predictions and extract bounding box values
def predict(image):
    result = learn.predict(image)
    if isinstance(result, tuple):
        prediction, bbox, *_ = result
    else:
        prediction, bbox = result, None
    return prediction, bbox

# Modify the main function to display bounding boxes
def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        prediction, bbox = predict(image)
        
        # Display the prediction
        st.write("Prediction:", prediction)
        
        # Draw bounding boxes on the image
        draw_bbox(image, bbox)

# Function to draw bounding boxes on the image
def draw_bbox(image, bbox):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(image))
    ax = plt.gca()
    
    # Convert bounding box tensor to numpy array
    if bbox is not None:
        bbox = bbox.detach().cpu().numpy()

        # Check if bbox is a single box or multiple boxes
        if len(bbox.shape) == 1:  # Single box
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            # Scale the coordinates to the image dimensions
            width, height = image.size
            x *= width
            y *= height
            w *= width
            h *= height
            print("Single box coordinates:", x, y, w, h)  # Debug statement
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y - 5, 'Turtle', color='red')
        else:  # Multiple boxes
            for box in bbox:
                x, y, w, h = box[0], box[1], box[2], box[3]
                # Scale the coordinates to the image dimensions
                width, height = image.size
                x *= width
                y *= height
                w *= width
                h *= height
                print("Multiple box coordinates:", x, y, w, h)  # Debug statement
                rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y - 5, 'Turtle', color='red')
            
    plt.axis('off')
    st.pyplot(plt)




if __name__ == "__main__":
    learn = load_model(model_path)
    main()
