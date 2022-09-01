Source Code for Real Time Damage Detection and Classifications
Source code for real time damage detection in Python programing language with use of
Open CV library, to run the application you need three file:
1- “yolov4-custom_best.weights” Obtained after the training of YOLOv4 object detector.
2- “yolov4-custom.cfg” Configuration file obtained from YOLOv4 author GitHub repo.
3- “obj.names” The damage category names file.
import cv2
import numpy as np
# 'yolov4-custom_best.weights' is learn able parameter of neural ne
twork, which we obtained after training the model.
net = cv2.dnn.readNet('yolov4-custom_best.weights',' yolov4-custom.
cfg ')
classes = []
with open(obj.names','r') as f:
classes = f.read().splitlines()
# For detecting from a Video (mp4)
#cap = cv2.VideoCapture('test4.mp4')
# For detecting from Video (webcame)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# For detecting from Video (Dev47Apps), it is an app which allows p
rogram to connect with a cell phone camera via wireless.
camera.cap = cv2.VideoCapture(1)
while True:
_,img = cap.read()
# print(classes)
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0
), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutPuts = net.forward(output_layers_names)
boxes = []
confidences = []
class_ids = []
for output in layerOutPuts:
for detection in output:
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence > 0.5:
center_x = int(detection[0] * width)
center_y = int(detection[1] * height)
w = int(detection[2] * width)
h = int(detection[3] * height)
x = int(center_x - w / 2)
y = int(center_y - h / 2)
boxes.append([x, y, w, h])
confidences.append(float(confidence))
class_ids.append(class_id)
print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))
# Loop for all object detected
if len(indexes) > 0:
for i in indexes.flatten():
x, y, w, h = boxes[i]
label = str(classes[class_ids[i]])
confidence = str(round(confidences[i], 2))
color = colors[i]
cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
cv2.putText(img, label + " " + confidence, (x, y + 20),
font, 2, (255, 255, 255), 2)
f = open('list.txt', 'a')
f.write(label)
f.write('\t')
f.write(str(confidence))
f.write('\n') f.close()
cv2.imshow('image', img)
key = cv2.waitKey(1)
if key == 27: break
cap.release()
cv2.destroyAllWindows()