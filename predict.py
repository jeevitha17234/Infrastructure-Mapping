import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.offline as pyo

WIDTH = 512
HEIGHT = 512

model = tf.keras.models.load_model("model/")
image_path = "image4.jpg"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = np.expand_dims(image, axis=0)
    return image

def visualize_predictions(input_image, model):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    
    # Make prediction using the provided model
    prediction = model.predict(input_image)
    
    # Resize the prediction to match the input image size
    prediction_resized = tf.image.resize(prediction, [HEIGHT, WIDTH]).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(input_image[0].squeeze())
    im = ax[1].imshow(prediction_resized[0].squeeze(), cmap=cmap)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    
    # Save plots as images
    plt.figure()
    plt.imshow(input_image[0].squeeze())
    plt.figure()
    plt.imshow(prediction_resized[0].squeeze(), cmap=cmap)
    plt.colorbar()
    plt.savefig("prediction.png")
    
    return prediction_resized

# Function to visualize point cloud
def visualize_point_cloud(depth_map, original_image):
    # Ensure depth map and original image have the same dimensions
    depth_map = np.flipud(depth_map.squeeze())
    original_image = np.flipud(original_image.squeeze())
    # Remove unnecessary dimensions
    h, w = depth_map.shape
    h_img, w_img, _ = original_image.shape

    if h != h_img or w != w_img:
        raise ValueError(f"Depth map dimensions ({h}, {w}) do not match input image dimensions ({h_img}, {w_img})")
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()

    colors = []
    for i in range(len(x)):
        hex_color = '#%02x%02x%02x' % tuple((original_image[y[i], x[i], :3] * 255).astype(int))
        colors.append(hex_color)

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(
        size=2,
        color=colors,
        opacity=0.8
    ))])

    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[0, w], title='X'),
        yaxis=dict(nticks=4, range=[0, h], title='Y'),
        zaxis=dict(nticks=4, range=[0, np.max(z)], title='Depth'),
    ))

    pyo.plot(fig, filename='point_cloud.html')
    
    
input = preprocess_image(image_path)
output = visualize_predictions(input, model)
# Call the function to visualize point cloud
visualize_point_cloud(output, input[0])
