import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np
import streamlit as st

st.write("Diabetes Detection Streamlit Practice")

df = pd.read_csv("C:/Users/Kevin/Desktop/ECS 171/Streamlit_practice/diabetes.csv")
st.subheader('Data Info:')
st.dataframe(df)
# Show statistics on the data
st.write(df.describe())
# show data as chart
chart = st.bar_chart(df)
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values

# Split data into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

# Get feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.number_input('insulin', 0)
    BMI = st.sidebar.number_input('BMI', 0)
    DPF = st.sidebar.slider('DPF', 0, 122, 72)
    age = st.sidebar.number_input('age', 0)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age,
        'uploaded_file': uploaded_file
    }
    # features = pd.DataFrame(user_data, index = [0])
    # return features
    return uploaded_file

user_input = get_user_input()
st.subheader('User Input: ')
# st.write(type(user_input))
st.write(user_input)
st.image(user_input)
image = st.image(user_input)
x = "C:/Users/Kevin/Pictures/" + user_input.name
st.write(x)
def map_image(x):
    return np.asarray(Image.open(x).resize((32,32)))
    
st.write(map_image(x))

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#Create and train model
# RFC = RandomForestClassifier()
# RFC.fit(X_train, Y_train)

# Show model
# st.subheader('Model Test Accuracy Score:')
# st.write(str(accuracy_score(Y_test, RFC.predict(X_test)) * 100) + '%')

# # Display prediction
# prediction = RFC.predict(user_input)
# st.subheader('Classification: ')
# st.write(prediction)