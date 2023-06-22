import cv2
from time import sleep
from PIL import Image

def main_app(name):
    # Chargement du classificateur de cascade pour la détection des visages
    face_cascade = cv2.CascadeClassifier('./donnee/haarcascade_frontalface_default.xml')
    
    # Chargement du classificateur entraîné pour la reconnaissance des visages
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"./donnee/classifiers/{name}_classifier.xml")
    
    # Ouverture de la capture vidéo à partir de la caméra
    cap = cv2.VideoCapture(0)
    
    pred = 0
    
    while True:
        ret, frame = cap.read()  # Lecture de l'image de la caméra
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion de l'image en niveaux de gris
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Détection des visages dans l'image
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]  # Région d'intérêt (ROI) de l'image en niveaux de gris
            
            id, confidence = recognizer.predict(roi_gray)  # Prédiction de l'identité du visage
            confidence = 100 - int(confidence)
            
            pred = 0
            
            if confidence > 50:
                pred += 1
                text = name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                pred += -1
                text = "UnknownFace"
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("image", frame)  # Affichage de l'image avec les visages détectés
        
        if cv2.waitKey(20) & 0xFF == ord('q'):  # Sortie de la boucle si la touche 'q' est pressée
            print(pred)
            
            if pred > 0:
                # Redimensionnement de l'image capturée
                dim = (124, 124)
                img = cv2.imread(f".\\donnee\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(f".\\donnee\\{name}\\50{name}.jpg", resized)
                
                # Composition de deux images pour afficher le résultat final
                Image1 = Image.open(f".\\2.png")
                Image1copy = Image1.copy()
                Image2 = Image.open(f".\\donnee\\{name}\\50{name}.jpg")
                Image2copy = Image2.copy()
                Image1copy.paste(Image2copy, (195, 114))
                Image1copy.save("end.png")
                frame = cv2.imread("end.png", 1)
                
                cv2.imshow("Résultat", frame)  # Affichage du résultat final
                cv2.waitKey(5000)  # Attendre 5 secondes
            break
    
    cap.release()  # Libération de la capture vidéo
    cv2.destroyAllWindows()  # Fermeture des fenêtres d'affichage
