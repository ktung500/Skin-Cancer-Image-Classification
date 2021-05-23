from keras.backend import exp
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from keras.models import load_model
import matplotlib.pyplot as plt

st.header("Skin Disease Detection")
st.image('./Pictures/home_image.png',  width = 400, caption="skin problem")

df = pd.read_csv('Data/HAM10000_metadata.csv',
                 names=['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
# remove row 0, because it's just the column names repeated
df.drop(index=0, inplace=True)
df = df.reset_index()

new_df = df[['age', 'sex', 'dx', 'localization']].copy()
# st.dataframe(new_df['localization'].value_counts())
st.dataframe(new_df)


# changing the type of age
new_df = new_df.astype({"age": np.float})

label_encoder = LabelEncoder()

label_encoder.fit(df['dx'])

# Create a new column named dx_encodings to hold our encoded diagnoses
df['dx_encodings'] = label_encoder.transform(df['dx'])


dx_dictionary = {
    'akiec': "actinic keratoses and intraepithelial carcinoma or Bowen's Disease",
    'bcc': 'basal cell carcinoma',
    'bkl': 'benign keratosis-like lesions',
    'df': 'dermatofibroma',
    'nv': 'melanocytic nevi',
    'vasc': 'pyogenic granulomas and hemorrhage',
    'mel': 'melanoma'}

new_df["dx"] = new_df["dx"].map(dx_dictionary)

st.sidebar.header('User Input: ')

st.write(new_df)
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
    # st.write(new_df.loc[((new_df["age"] >= (user_input["age"] - 10))
    #                      & (new_df["age"] <= (user_input["age"] + 10)))])

    age_sex_df = new_df.loc[(new_df["sex"] == user_input["sex"]) & ((new_df["age"] >= (user_input["age"] - 10))
                                                                    & (new_df["age"] <= (user_input["age"] + 10)))]

    # labels = age_sex_df['dx'].unique()
    # st.write(labels)
    # sizes = age_sex_df['dx'].value_counts()
    st.subheader(
        "The kinds of skin problems that you are most likely to have given your age and sex.")
    st.bar_chart(age_sex_df['dx'].value_counts())

    if user_input["uploaded_file"]:
        st.write("Using the image that you gave us.")

        x = "./Data/HAM10000_images_part_1/" + user_input["uploaded_file"].name

        def map_image(x):
            IMAGE_SIZE = 32
            return np.asarray(Image.open(x).resize((IMAGE_SIZE, IMAGE_SIZE)))

        st.write("Loading the model")
        model = load_model('CNN_skin_lesion_model')
        st.write("Model is loaded")

        numpy_image_data = map_image(x)

        # Normalize the data first before predicting

        numpy_image_data = numpy_image_data / 255.

        numpy_image_data = [np.array(numpy_image_data)]

        st.write("Performing prediction on the image that was provided")
        y_pred = model.predict(np.array(numpy_image_data))
        y_pred = np.squeeze(y_pred)
        index = np.where(y_pred == y_pred.max())

        st.write("Your skin problem is classified as:", dx_dictionary[label_encoder.inverse_transform(index[0])[0]])

        st.write("What percentage of skin disease matches your classification?")

        # show the frequency of different types of disesases on a global scale
        diseases = new_df['dx'].value_counts()
        diseases_name= new_df['dx'].value_counts().index

        explode = []
        for i in range(len(diseases_name) -1):
            if diseases_name[i] == dx_dictionary[label_encoder.inverse_transform(index[0])[0]]:
                explode.append(0.5)
            explode.append(0)
        plt.pie(diseases, labels=diseases_name, radius=2, autopct='%.1f%%', shadow=False, explode=explode)
        # don't want to see warning :)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot()

        # show the pie chart with the frequencies of disease in male, female, and unknown
        only_disease = new_df.loc[(new_df["dx"] == dx_dictionary[label_encoder.inverse_transform(index[0])[0]])]
        st.write(only_disease)
        st.write("Frequencey of gender that has ", dx_dictionary[label_encoder.inverse_transform(index[0])[0]], " disease")
        only_disease_sex = only_disease['sex'].value_counts()
        only_disease_sex_name= only_disease['sex'].value_counts().index
        plt.pie(only_disease_sex, labels=only_disease_sex_name, radius=2, autopct='%.1f%%', shadow=False)
        st.pyplot()

        # show how likely is this disease to be on the body parts that the user chose
        st.write("How likely is it to get this disease on the ", user_input["body"], "?")
        body_place = only_disease['localization'].value_counts()
        body_place_name = only_disease['localization'].value_counts().index
        explode2 = []
        for i in range(len(body_place_name) -1):
            if body_place_name[i] == user_input["body"]:
                explode2.append(0.5)
            explode2.append(0)
        plt.pie(body_place, labels=body_place_name, radius=2, autopct='%.1f%%', shadow=False, explode=explode2)
        st.pyplot()
        
        disease_to_website = {
            "actinic keratoses and intraepithelial carcinoma or Bowen's Disease": "https://www.uhb.nhs.uk/Downloads/pdf/PiActinicKeratosesAndBowensDisease.pdf",
            'basal cell carcinoma': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer/about.html',
            'benign keratosis-like lesions': 'https://www.ncbi.nlm.nih.gov/books/NBK545285/',
            'dermatofibroma': 'https://www.dermnetnz.org/topics/dermatofibroma/',
            'melanocytic nevi': 'https://www.dermnetnz.org/topics/melanocytic-naevus/',
            'pyogenic granulomas and hemorrhage': 'https://www.dermnetnz.org/topics/pyogenic-granuloma/',
            'melanoma':'https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884'
            }

        # give them resources to get everything checked
        link = disease_to_website[dx_dictionary[label_encoder.inverse_transform(index[0])[0]]]
        st.markdown(f"If you want to learn more about your diagonosis, then [click here] ({link})")
    else:
        st.subheader("If you want to diagnosis your disease then please upload your skin image in the side bar. Thank you!")
