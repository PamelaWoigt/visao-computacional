import cv2
import numpy as np

# Carregar a rede YOLOv3
net = cv2.dnn.readNet("yolov3.cfg")

# Carregar nomes das classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definir camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar imagem
image = cv2.imread("GATOS.jpeg")
height, width, channels = image.shape

# Preparar a imagem para a detecção
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Analisar as detecções
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Obter coordenadas da caixa delimitadora
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar Non-Max Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhar caixas delimitadoras
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        if label == "cat":
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), font, 1, (0, 255, 0), 2)

# Mostrar a imagem
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
