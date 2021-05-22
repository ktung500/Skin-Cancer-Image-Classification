import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
import streamlit as st
from keras.models import load_model
import matplotlib.pyplot as plt

st.write("Skin Disease Detection")

df = pd.read_csv('Data/HAM10000_metadata.csv',
                 names=['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
# remove row 0, because it's just the column names repeated
df.drop(index=0, inplace=True)
df = df.reset_index()

new_df = df[['age', 'sex', 'localization']].copy()
# st.dataframe(new_df['localization'].value_counts())
# st.dataframe(new_df)

st.image('./Pictures/home_image.png',  width = 400)

# changing the type of age
new_df = new_df.astype({"age": np.float})

label_encoder = LabelEncoder()

label_encoder.fit(df['dx'])

# Create a new column named dx_encodings to hold our encoded diagnoses
df['dx_encodings'] = label_encoder.transform(df['dx'])

df.head(5)


def get_user_input():
    age = st.sidebar.slider('Age', 0, 100)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    body = st.sidebar.selectbox("On what part of the body is the pigment in?", [
        'Back', 'Lower Extremity', 'Trunk', 'Upper Extremity', 'Abdomen',
        'Face', 'Chest', 'Foot', 'Hand', 'Ear', 'Genital', 'Acral', 'Unknown'])
    uploaded_file = st.sidebar.file_uploader(
        "Please insert an image that you want to classify", type="jpg")
    user_input = {"age": age, "sex": sex.lower(),
                  "body": body.lower(), "uploaded_file": uploaded_file}
    return user_input


user_input = get_user_input()

if user_input is not None:
    st.subheader('User Input: ')
    st.write(new_df.columns)
    if user_input["uploaded_file"]:
        st.image(user_input["uploaded_file"])
    st.write(type(user_input["age"]))

    # st.write(new_df.loc[((new_df["age"] >= (user_input["age"] - 10))
    #                      & (new_df["age"] <= (user_input["age"] + 10)))])

    age_sex_df = new_df.loc[(new_df["sex"] == user_input["sex"]) & ((new_df["age"] >= (user_input["age"] - 10))
                                                                    & (new_df["age"] <= (user_input["age"] + 10)))]

    # st.dataframe(age_sex_df)

    # st.write(age_sex_df.shape)
    # st.write(age_sex_df['localization'].value_counts())
    # st.write(age_sex_df['localization'].unique())
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = age_sex_df['localization'].unique()
    st.write(labels)
    sizes = age_sex_df['localization'].value_counts()
    st.write(
        "The kinds of skin problems that you are most likely to have given your age and sex.")
    st.bar_chart(age_sex_df['localization'].value_counts())

    # explode = (0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax1.axis('equal')

    # st.pyplot(fig1)

    # x = "./Data/HAM10000_images_part_1/" + user_input.name

    # def map_image(x):
    #     IMAGE_SIZE = 32
    #     return np.asarray(Image.open(x).resize((IMAGE_SIZE, IMAGE_SIZE)))

    # model = load_model('CNN_skin_lesion_model')

    # st.write(x)

    # st.write(map_image(x))

    # numpy_image_data = map_image(x)

    # st.write(numpy_image_data.shape)

    # # Normalize the data first before predicting

    # numpy_image_data = numpy_image_data / 255.

    # numpy_image_data = [np.array(numpy_image_data)]

    # y_pred = model.predict(np.array(numpy_image_data))
    # y_pred = np.squeeze(y_pred)
    # index = np.where(y_pred == y_pred.max())

    # print(f"It is {label_encoder.inverse_transform(index[0])[0]}")

    # dx_dictionary = {
    #     'akiec': "actinic keratoses and intraepithelial carcinoma or Bowen's Disease",
    #     'bcc': 'basal cell carcinoma',
    #     'bkl': 'benign keratosis-like lesions',
    #     'df': 'dermatofibroma',
    #     'nv': 'melanocytic nevi',
    #     'vasc': 'pyogenic granulomas and hemorrhage',
    #     'mel': 'melanoma'}

    # st.write(dx_dictionary[label_encoder.inverse_transform(index[0])[0]])

    # print("You got cancer :)")

else:
    st.write("LOL")
