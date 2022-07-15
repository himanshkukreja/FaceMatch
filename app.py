from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

config = read_yaml("config/config.yaml")
params = read_yaml("params.yaml")

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
upload_path= os.path.join(artifacts_dir,upload_image_dir)

#pickle_format_data_dir
pickle_format_data_dir = artifacts['pickle_format_data']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir,pickle_format_data_dir)
pickle_file= os.path.join(raw_local_dir_path,img_pickle_file_name)

# containes all the file names of original databse images before extractions/applying model 
filenames = pickle.load(open(pickle_file,'rb'))


#feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name= artifacts['extracted_features_name']

feature_extraction_path= os.path.join(artifacts_dir,feature_extraction_dir)
feature_name = os.path.join(feature_extraction_path,extracted_features_name)

# contains all the extracted features of databse images in single 2D array where eache row contains 2048 values and having more than 8000 rows(no. of images)
feature_list = pickle.load(open(feature_name,'rb'))


#params
model_name=params['base']['BASE_MODEL']
include_tops=params['base']['include_top']
poolings=params['base']['pooling']

detector = MTCNN()
model = VGGFace(model = model_name, include_top=include_tops,input_shape=(224,224,3),pooling=poolings)


#save upload image
def save_upload_image(uploadimage):
    try:
        create_directory(dirs=[upload_path])
        with open(os.path.join(upload_path,uploadimage.name),'wb') as f:
            f.write(uploadimage.getbuffer())
        return True
    except:
        return False

#Extract feature from uploaded image
def extract_feature(img_path,model,detector):
    
    #detect face using MTCNN face detector
    img=cv2.imread(img_path)
    result = detector.detect_faces(img)
    x,y,width,height = result[0]['box']
    face = img[x:x+width,y:y+height]

    #Extracting features using resnet50 model
    image = Image.fromarray(face)
    image = image.resize((224,224))
    
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    #Expand the image array to add the extra dimention of batch
    expanded_img = np.expand_dims(face_array,axis=0)

    #preprocess the expanded image
    preprocess_img = preprocess_input(expanded_img)

    #passing the preprocessed image to our model which after few layers of convolutions and pooling gives the image in same format as we processed the images of our database of over 8000 images for comparing
    result = model.predict(preprocess_img).flatten()

    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])
    index_pos = sorted(list(enumerate(similarity)),reverse=True, key=lambda x: x[1])[0][0]
    return index_pos



print(len(feature_list))

#streamlit
st.title('Check Your Doppleganger')

uploadimage = st.file_uploader('chose an image')

if uploadimage is not None:
    #save the image
    if save_upload_image(uploadimage):
        #load image 
        display_image=Image.open(uploadimage)

        #extract_feature
        features = extract_feature(os.path.join(upload_path,uploadimage.name),model,detector)

        #recommending or comparing/measuring the cosine similarity between all database and uploaded imag features

        indexpos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[indexpos].split('\\')[1].split('_'))

        #for displaying the image of resulted actor
        col1,col2 = st.columns(2)
        with col1:
            st.header('You')
            st.image(display_image)

        with col2:
            st.header('You Look Likes '+predicted_actor)
            st.image(filenames[indexpos],width =300)







