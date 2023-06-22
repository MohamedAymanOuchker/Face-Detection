import cv2
import os

def start_capture(name):
    # Définition du chemin de destination
    path = "./donnee/" + name
    
    # Compteur du nombre d'images capturées
    num_of_images = 0
    
    # Chargement du classificateur de cascade pour la détection des visages
    detector = cv2.CascadeClassifier("./donnee/haarcascade_frontalface_default.xml")
    
    try:
        os.makedirs(path)  # Création du répertoire de destination s'il n'existe pas
    except:
        print('Répertoire déjà existant')
        
    # Capture vidéo à partir de la caméra
    vid = cv2.VideoCapture(0)
    
    while True:
        ret, img = vid.read()  # Lecture de l'image de la caméra
        
        new_img = None
        
        # Conversion de l'image en niveaux de gris pour la détection des visages
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages dans l'image
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        
        # Parcours de tous les visages détectés
        for x, y, w, h in face:
            # Dessin d'un rectangle autour du visage détecté
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            
            # Affichage du texte "Face Detected" au-dessus du visage détecté
            cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            
            # Affichage du nombre d'images capturées en-dessous du visage détecté
            cv2.putText(img, str(str(num_of_images)+" images capturées"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            
            # Extraction du visage de l'image complète
            new_img = img[y:y+h, x:x+w]
        
        # Affichage de l'image avec les visages détectés
        cv2.imshow("Détection de visages", img)
        
        key = cv2.waitKey(1) & 0xFF
        
        try:
            # Sauvegarde de l'image extraite dans le répertoire de destination
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except:
            pass
        
        # Sortie de la boucle si la touche 'q' ou 'ESC' est pressée ou si le nombre maximal d'images est atteint
        if key == ord("q") or key == 27 or num_of_images > 310:
            break
    
    cv2.destroyAllWindows()
    return num_of_images
