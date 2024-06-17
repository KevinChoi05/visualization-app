import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, concatenate, BatchNormalization, SeparableConv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="D-Prime Prediction App", layout="centered", initial_sidebar_state="auto")

@st.cache_data()
def create_and_load_model(weights_path):
    def create_image_branch(input_shape):
        inputs = Input(shape=input_shape, name='image_input')
        x = SeparableConv2D(256, kernel_size=(3, 3), activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = SeparableConv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
        return inputs, x

    def create_metadata_branch(input_shape):
        inputs = Input(shape=input_shape, name='metadata_input')
        x = Dense(48, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        return inputs, x

    image_input, image_branch = create_image_branch((224, 224, 3))
    metadata_input, metadata_branch = create_metadata_branch((4,))
    combined = concatenate([image_branch, metadata_branch])

    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    outputs = outputs * 3

    model = Model(inputs=[image_input, metadata_input], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='mse', metrics=['mse'])
    model.load_weights(weights_path)
    return model

# Assuming your model weights are stored in the same directory as this script
model_path = 'my_model_weights2.h5'
model = create_and_load_model(model_path)

st.title('🔮 D-Prime Prediction App')
st.markdown('### Upload your image and get insights on the d-prime value.')

# Instructions at the top
st.markdown("""
## Instructions
1. Upload an image using the uploader below.
2. Set the image metadata options.
3. Click on "Predict" to get the d-prime value.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption='Uploaded Image.', use_column_width=True)

col1, col2 = st.columns(2)
with col1:
    human_recognizable = st.selectbox('Is there a human recognizable object?', ('Yes', 'No'))
    black_white = st.selectbox('Is the image black and white?', ('Yes', 'No'))
with col2:
    distinct_colors = st.number_input('Number of distinct colors', min_value=0, max_value=10, step=1)
    human_depiction = st.selectbox('Is there a human depiction?', ('Yes', 'No'))

if st.button('Predict'):
    if uploaded_file is not None:
        img = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        human_recognizable = 1 if human_recognizable == 'Yes' else 0
        black_white = 1 if black_white == 'Yes' else 0
        human_depiction = 1 if human_depiction == 'Yes' else 0
        
        metadata_input = np.array([[human_recognizable, distinct_colors, black_white, human_depiction]])

        prediction = model.predict([img_array, metadata_input])
        d_prime_predicted = prediction[0][0]

        # Visualization of the prediction using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = d_prime_predicted,
            title = {'text': "D-Prime Value"},
            gauge = {'axis': {'range': [None, 3]},
                     'bar': {'color': "lightblue"},
                     'steps' : [
                         {'range': [0, 1], 'color': "lightgray"},
                         {'range': [1, 2], 'color': "gray"},
                         {'range': [2, 3], 'color': "darkgray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2.5}}))

        st.plotly_chart(fig)

        # Explanation of the d-prime value
        st.markdown(f"### Predicted D-Prime Value: {d_prime_predicted:.3f}")
        st.markdown("""
        The d-prime value is a statistical measure used to quantify the memorability score.
        In the context of this model, a higher d-prime value suggests a higher chance of the visualization being remembered.
        """)
    else:
        st.error('Please upload an image to get a prediction.')
else:
    st.info('Click the predict button after uploading an image and setting metadata options.')

# Additional information about the app at the bottom
st.markdown('---')
st.markdown("""
## About
This app uses a deep learning model to predict the d-prime value based on visual features and metadata. 
It is designed to be a demonstration of integrating machine learning models into interactive web applications.
""")
