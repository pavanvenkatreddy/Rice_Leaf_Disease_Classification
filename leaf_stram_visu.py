import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Load the saved model
loaded_model = load_model('/Users/pavanvenkatreddy/Downloads/my_model (1).h5')  # Update the path

# Define custom class labels
custom_class_labels = [
    'Healthy',
    'Mild Bacterial blight',
    'Mild Blast',
    'Mild Brownspot',
    'Mild Tungro',
    'Severe Bacterial blight',
    'Severe Blast',
    'Severe Brownspot',
    'Severe Tungro'
]

# Define image dimensions
img_height, img_width = 300, 300

# Function to preprocess image
def preprocess_image(img_file):
    img = image.load_img(img_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0
    return img_array

# Function to visualize intermediate activations
def visualize_intermediate_activations(img_path, layer_index):
    # Load and preprocess the image
    img_tensor = preprocess_image(img_path)

    # Extract activations of the specified layer
    layer_output = loaded_model.layers[layer_index].output
    activation_model = Model(inputs=loaded_model.input, outputs=layer_output)
    intermediate_activations = activation_model.predict(img_tensor)

    # Plot the activations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    activation = intermediate_activations[0]  # Get the only activation array
    for j in range(3):  # Iterate over each channel
        axes[j].imshow(activation[:, :, j], cmap='viridis')
        axes[j].set_title(f'Channel {j+1}')
        axes[j].axis('off')

    # Save the figure as a PNG file
    plt.savefig('activations.png')

# Streamlit app
st.title('Rice Leaf Disease Detection and Activations Visualizer')

# Display model architecture
st.subheader('Model Architecture')
with st.expander("Show Model Summary"):
    loaded_model.summary(print_fn=lambda x: st.text(x))


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    #st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    img_array = preprocess_image(uploaded_file)
    predictions = loaded_model.predict(img_array)
    predicted_label = np.argmax(predictions[0])
    predicted_class = custom_class_labels[predicted_label]

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    #st.write("Classifying...")
    st.write(f"Predicted Class: {predicted_class}")

    # Slider to select the layer index
    layer_index = st.slider('Select Layer Index', min_value=0, max_value=17, value=3)

    # Button to trigger visualization
    #if st.button('Visualize Intermediate Activations'):
    visualize_intermediate_activations(uploaded_file, layer_index)
        # Display the saved image
    st.image('activations.png', caption='Intermediate Activations', use_column_width=True)
