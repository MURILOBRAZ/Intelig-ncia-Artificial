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
            # Extrair informações de caixa delimitadora do rosto
            caixa_delimitadora = rosto.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(caixa_delimitadora.xmin * iw), int(caixa_delimitadora.ymin * ih), \
                         int(caixa_delimitadora.width * iw), int(caixa_delimitadora.height * ih)
            
            # Calcular a posição do texto
            text_x = x + int(w / 2) - 50
            text_y = y + h + 20

            # Desenhar caixa delimitadora em torno do rosto
            desenho.draw_detection(frame, rosto)
            
            # Adicionar texto com nome fictício
            cv2.putText(frame, "Nome: Murilo", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Reconhecimento Facial", frame)

    #fecha com o ESC
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
    