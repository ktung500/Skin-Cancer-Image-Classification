import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
import streamlit as st
from keras.models import load_model
# import matplotlib as plt

st.write("Skin Disease Detection")

df = pd.read_csv('Data/HAM10000_metadata.csv',
                 names=['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
# remove row 0, because it's just the column names repeated
df.drop(index=0, inplace=True)
df = df.reset_index()
df.head()
st.dataframe(df['localization'].value_counts())
st.dataframe(df)

label_encoder = LabelEncoder()

label_encoder.fit(df['dx'])

# Create a new column named dx_encodings to hold our encoded diagnoses
df['dx_encodings'] = label_encoder.transform(df['dx'])

df.head(5)


def get_user_input():
    age = st.sidebar.slider('Age', 0, 100)
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    body = st.sidebar.selectbox("On what part of the body is the pigment in?", [
                    'back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 
                    'face', 'chest', 'foot', 'hand', 'ear', 'genital', 'acral', 'unknown'])
    
    # where = st.text_input
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    return uploaded_file


user_input = get_user_input()

if user_input is not None:
    st.subheader('User Input: ')
    st.write(user_input)
    # st.image(user_input)

    x = "./Data/HAM10000_images_part_1/" + user_input.name

    def map_image(x):
        IMAGE_SIZE = 32
        return np.asarray(Image.open(x).resize((IMAGE_SIZE, IMAGE_SIZE)))

    model = load_model('CNN_skin_lesion_model')

    st.write(x)

    st.write(map_image(x))

    numpy_image_data = map_image(x)

    st.write(numpy_image_data.shape)

    # Normalize the data first before predicting

    numpy_image_data = numpy_image_data / 255.

    numpy_image_data = [np.array(numpy_image_data)]

    y_pred = model.predict(np.array(numpy_image_data))
    y_pred = np.squeeze(y_pred)
    index = np.where(y_pred == y_pred.max())

    print(f"It is {label_encoder.inverse_transform(index[0])[0]}")

    dx_dictionary = {
        'akiec': "actinic keratoses and intraepithelial carcinoma or Bowen's Disease",
        'bcc': 'basal cell carcinoma',
        'bkl': 'benign keratosis-like lesions',
        'df': 'dermatofibroma',
        'nv': 'melanocytic nevi',
        'vasc': 'pyogenic granulomas and hemorrhage',
        'mel': 'melanoma'}

    st.write(dx_dictionary[label_encoder.inverse_transform(index[0])[0]])

    print("You got cancer :)")

else:
    st.write("LOL")