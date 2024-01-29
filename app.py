import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
model = load_model('assets/model/model.h5')

# Load the tokenizer used during training
with open('assets/model/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Dictionary mapping categories to images
category_images = {
    'Anis-Imin': 'assets/images/1.gif',
    'Prabowo-Gibran': 'assets/images/2.gif',
    'Ganjar-Mahfud': 'assets/images/3.gif'
}

# Streamlit app
def main():
    st.title("Aplikasi Klasifikasi Berita Pemilu 2024")

    # User input text box
    user_input = st.text_area("Masukkan berita pemilu:", "")

    if st.button("Klasifikasi Berita!"):
        # Make prediction
        prediction = predict_news_category(user_input)

        # Display the results
        st.subheader("Nilai:")
        display_prediction_table(prediction)

        # Display image based on max_category
        display_category_image(prediction)

def predict_news_category(user_input):
    # Preprocess user input
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=8958)

    # Make prediction
    prediction = model.predict(user_input_padded)

    return prediction[0]

def display_prediction_table(prediction):
    # Create a DataFrame with the predicted values
    categories = ['Anis-Imin', 'Prabowo-Gibran', 'Ganjar-Mahfud']
    df_prediction = pd.DataFrame({'Category': categories, 'Prediction': prediction})

    # Find the category with the highest prediction value
    max_category = df_prediction.loc[df_prediction['Prediction'].idxmax(), 'Category']

    # Display the prediction table
    st.table(df_prediction)

    # Display the result based on the highest prediction
    st.subheader(f"Hasil Klasifikasi: {max_category}")

def display_category_image(prediction):
    # Find the category with the highest prediction value
    max_category_index = np.argmax(prediction)
    categories = ['Anis-Imin', 'Prabowo-Gibran', 'Ganjar-Mahfud']
    max_category = categories[max_category_index]

    # Get the image filename based on the max_category value
    image_filename = category_images.get(max_category, 'assets/images/default-image.jpg')  # Replace with the default image filename

    # Display the image
    st.image(image_filename, caption=f"Image for {max_category}", use_column_width=True)

if __name__ == "__main__":
    main()
