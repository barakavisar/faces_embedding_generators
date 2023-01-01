import tensorflow
import os
import pandas as pd
import numpy as np
from numpy import asarray
import cv2
from PIL import Image
#from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial import distance as dist

def generate_embeddings():
    # the models
    models = ["VGG-Face", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
    face_model = DeepFace.build_model('Facenet')
    embeddings = []
    names = []

    path = 'data/your_path_to_crops_directory'
    images_list = os.listdir(path)
    print('images_list', images_list[0:3])
    for j, filename in enumerate(images_list):

        img = cv2.imread(path + '/' + filename)
        # img = cv2.resize(img, (112, 112), cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        embedding = face_model.predict(img)
        embedding = embedding.astype('float32')
        names.append(filename)

        embeddings.append(embedding)
        print('j', j)

    len_result = len(embeddings)
    print('len_embeddings', len_result)

    embeddings = np.array(embeddings, dtype=float).reshape(len_result, 128)
    names = np.array(names, dtype=str)

    embeddings = pd.DataFrame(embeddings, dtype=float)
    names = pd.DataFrame(names, dtype=str)
    names.columns = ['filename']
    result_embeddings = pd.concat((names, embeddings), axis=1)
    result_embeddings.to_csv('results/embeddings.....csv')

generate_embeddings()


