import cv2 #opencv
import mediapipe as mp

#inicializar opencv e mediapipe
webcam = cv2.VideoCapture(1)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador_leu, frame = webcam.read()
    if not verificador_leu:
        break

    lista_rostos = reconhecedor_rostos.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)
    
    cv2.imshow("Reconhecimento Facial", frame)

    #fecha com o ESC
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()