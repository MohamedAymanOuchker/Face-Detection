import numpy as np
from PIL import Image
import os
import cv2

def train_classifer(name):
    # Définition du chemin du répertoire contenant les images d'entraînement
    path = os.path.join(os.getcwd() + "/donnee/" + name + "/")

    faces = []  # Liste pour stocker les visages
    ids = []  # Liste pour stocker les identifiants des visages
    labels = []  # Liste pour stocker les libellés des visages (facultatif)
    pictures = {}  # Dictionnaire pour stocker les fichiers d'images

    # Parcours de l'arborescence des répertoires pour obtenir la liste des fichiers d'images
    for root, dirs, files in os.walk(path):
        pictures = files

    # Parcours de chaque image
    for pic in pictures:
        imgpath = path + pic  # Chemin de l'image
        img = Image.open(imgpath).convert('L')  # Ouverture de l'image en niveaux de gris
        imageNp = np.array(img, 'uint8')  # Conversion de l'image en tableau numpy
        id = int(pic.split(name)[0])  # Extraction de l'identifiant du visage à partir du nom du fichier
        faces.append(imageNp)  # Ajout du visage à la liste des visages
        ids.append(id)  # Ajout de l'identifiant du visage à la liste des identifiants

    # Conversion des listes en tableaux numpy
    ids = np.array(ids)

    # Création et entraînement du classificateur LBPH (Local Binary Patterns Histogram)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Sauvegarde du classificateur entraîné dans un fichier XML
    clf.write("./donnee/classifiers/" + name + "_classifier.xml")
