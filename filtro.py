import numpy as np
import cv2
import matplotlib.pyplot as plt

def anonymize_face_pixelate(image, hInit, wInit, hEnd, wEnd, blocks=30):
	#Separa a imagem e cria um array com todos os pontos que precisam ser preenchidos
	xSteps = np.linspace(hInit, hEnd, blocks + 1, dtype="int")
	ySteps = np.linspace(wInit, wEnd, blocks + 1, dtype="int")
	#Percorre todos os pontos que precisam ser desfocados e desfoca
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# Pega as coordenadas do bloco corrente
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			# coloca sépia onde tem o rosto
			cv2.rectangle(image, (startX, startY), (endX, endY), (255 - B, 255 - G, 255 - R), -1)
	# return the pixelated blurred image
	return image

cap = cv2.VideoCapture(0)
width = 1295
height = 800
dim = (width, height)


while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)	

  #carrega face detection    
  haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  faces_rects = haar_cascade_face.detectMultiScale(frame, 1.2, 5)
 
  for (x,y,w,h) in faces_rects:
		#coloca retangulo onde encontra rosto com ajuste para pegar uma área maior relativo a toda a cabeça
    cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (0, 255, 0), 2)
		#deixa o rosto anônimo
    frame = anonymize_face_pixelate(frame, x-50, y-50, x+w+50, y+h+50)
		
  # Display the resulting frame

  # convert img to grayscale
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
