import cv2
import mediapipe as mp

#vincular webcam
webcam = cv2.VideoCapture(1)
reconhecimento_de_maos = mp.solutions.hands
desenhomp = mp.solutions.drawing_utils
maos = reconhecimento_de_maos.Hands()

if webcam.isOpened():

    estado, frame = webcam.read()
    while estado:

        estado, frame = webcam.read()
        
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lista_maos = maos.process(frameRGB)
        if lista_maos.multi_hand_landmarks:

            for mao in lista_maos.multi_hand_landmarks:
                print(mao.landmark)
                desenhomp.draw_landmarks(frame, mao, reconhecimento_de_maos.HAND_CONNECTIONS)

        cv2.imshow("Reconhecimento De Mãos", frame)

        tecla = cv2.waitKey(2)

        if tecla == 27:
            break
        
webcam.release()
cv2.destroyAllWindows()