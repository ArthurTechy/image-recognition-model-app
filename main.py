import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .block-container {
        padding-top: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('cifar10_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Cache the image preprocessing
@st.cache_data(hash_funcs={Image.Image: lambda x: hash(x.tobytes())})
def preprocess_image(image):
    # Resize image to 32x32
    image = image.resize((32, 32))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Cache sample images loading
@st.cache_data
def load_sample_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test

def main():
    # Initialize the model
    model = load_model()
    if model is None:
        st.error("Please ensure 'cifar10_model.h5' is in the same directory as this script.")
        return

    # Sidebar
    with st.sidebar:
        st.title("🖼️ CIFAR-10 Classifier")
        st.markdown("### Navigation")
        page = st.radio(
            "Choose a page",
            ["Image Classification", "Model Performance", "Sample Predictions"]
        )
        
        st.markdown("### About")
        st.write("""
        This app uses a pre-trained CNN model to classify images into 10 categories:
        airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
        """)

    # Main content
    if page == "Image Classification":
        st.header("Image Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                
                if st.button("Classify Image"):
                    with st.spinner("Processing..."):
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        predictions = model.predict(processed_image)
                        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                     'dog', 'frog', 'horse', 'ship', 'truck']
                        
                        # Get top prediction
                        top_pred_idx = np.argmax(predictions[0])
                        top_pred_class = class_names[top_pred_idx]
                        top_pred_conf = predictions[0][top_pred_idx]

        with col2:
            if uploaded_file is not None and 'predictions' in locals():
                st.markdown("### Prediction Results")
                
                # Display top prediction
                st.markdown(f"""
                <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                    <h3 style='margin: 0;'>Top Prediction: {top_pred_class.title()}</h3>
                    <p style='margin: 0;'>Confidence: {top_pred_conf*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display bar chart of all predictions
                fig = go.Figure(data=[go.Bar(
                    x=[pred*100 for pred in predictions[0]],
                    y=[name.title() for name in class_names],
                    orientation='h'
                )])
                
                fig.update_layout(
                    title="Prediction Confidence for All Classes",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Class",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Load test data
        x_test, y_test = load_sample_data()
        
        # Make predictions on test set
        test_samples = 1000  # Limit to first 1000 samples for speed
        predictions = model.predict(x_test[:test_samples] / 255.0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = y_test[:test_samples].flatten()
        
        # Create confusion matrix
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        confusion_mtx = tf.math.confusion_matrix(true_classes, pred_classes)
        
        # Plot confusion matrix
        fig = px.imshow(confusion_mtx,
                       labels=dict(x="Predicted label", y="True label"),
                       x=class_names,
                       y=class_names,
                       title="Confusion Matrix")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display accuracy
        accuracy = np.mean(pred_classes == true_classes)
        st.metric("Test Set Accuracy", f"{accuracy*100:.2f}%")

    elif page == "Sample Predictions":
        st.header("Sample Predictions")
        
        # Load test data
        x_test, y_test = load_sample_data()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Random sample selection
        if st.button("Show New Random Samples"):
            sample_indices = np.random.choice(len(x_test), 9, replace=False)
            
            # Create 3x3 grid of sample images
            cols = st.columns(3)
            for idx, sample_idx in enumerate(sample_indices):
                col = cols[idx % 3]
                with col:
                    # Display image
                    img = x_test[sample_idx]
                    st.image(img, caption=f"True: {class_names[y_test[sample_idx][0]]}")
                    
                    # Make prediction
                    pred = model.predict(np.expand_dims(img, 0) / 255.0)
                    pred_class = class_names[np.argmax(pred[0])]
                    pred_conf = np.max(pred[0]) * 100
                    
                    st.markdown(f"""
                    <div style='padding: 0.5rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <p style='margin: 0;'>Predicted: {pred_class}</p>
                        <p style='margin: 0;'>Confidence: {pred_conf:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
