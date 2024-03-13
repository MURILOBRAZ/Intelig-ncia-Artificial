import mediapipe as mp
import cv2

# Inicialize o mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicialize o detector de rosto
face_detection = mp_face_detection.FaceDetection()

# Carregue uma imagem de exemplo
image = cv2.imread("pessoas.jpg")

# Converta a imagem para o formato RGB (MediaPipe usa RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detecte rostos na imagem
results = face_detection.process(image_rgb)

# Desenhe caixas delimitadoras ao redor dos rostos detectados
if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

# Exiba a imagem com as caixas delimitadoras desenhadas
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
