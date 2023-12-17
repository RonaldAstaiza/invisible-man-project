# Importamos las librerias
from ultralytics import YOLO
import cv2
from imutils.video import FPS
import numpy as np
#480,640
size=480
# Leer nuestro modelo
print("[INFO] loading YOLOV8 from disk...")
model = YOLO("best480-2.pt")
kernel = np.ones((5,5),np.uint8)
# Realizar VideoCaptura
cap = cv2.VideoCapture(0)
print('iniciando toma de Background')
for _ in range(10):
    _,bg = cap.read()
# Bucle
print('finalizada toma de Background')
fps = FPS().start()
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()
    resized = cv2.resize(frame,(size,size),interpolation = cv2.INTER_AREA)
    bg= cv2.resize(bg,(size,size),interpolation = cv2.INTER_AREA)
    resultados = model.predict(resized)[0]
    cv2.imshow("Original", frame)
    try:
        mask=resultados.masks.data.tolist()
        
    except AttributeError:
        continue
    bwmask = np.array(mask,dtype=np.uint8) * 255
    bwmask = np.reshape(bwmask,np.array(mask).shape)
    #bwmask = cv2.dilate(bwmask,kernel,iterations=1)
    resized[np.where(bwmask[0]==255)] = bg[np.where(bwmask[0]==255)]


    # Mostramos nuestros fotogramas
    cv2.imshow("SEGMENTACION", resized)
   

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break
    fps.update()

        

cap.release()
cv2.destroyAllWindows()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))