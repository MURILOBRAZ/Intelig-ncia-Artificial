import cv2
from deepface import DeepFace

imagem = cv2.imread("triste.jpg")

resultado = DeepFace.analyze(imagem, actions=("age", "gender", "race", "emotion"))

print(resultado)