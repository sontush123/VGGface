import os
import streamlit as st
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
# Function to load image
detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos



# Streamlit app
st.title("Which bollywood celebrity are you?")

# File uploader
image_file = st.file_uploader("Upload an Image", type=['png', 'jpeg', 'jpg'])

if image_file is not None:
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    st.write(file_details)

    # Display the uploaded image
    img = Image.open(image_file)
    #st.image(img, width=250)

    # Save the uploaded image to "tempDir"
    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = os.path.join(temp_dir, image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())

    st.success("Saved File to tempDir")

    features = extract_features(os.path.join(temp_dir,image_file.name),model,detector)
    index_pos = recommend(feature_list,features)
    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
    col1,col2 = st.columns(2)
    with col1:
        st.header('Your uploaded image')
        st.image(img)
    with col2:
        st.header("Seems like " + predicted_actor)
        st.image(filenames[index_pos],width=300)




