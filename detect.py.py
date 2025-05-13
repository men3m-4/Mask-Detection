print("===> Script started")

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model

print("Loading face detector...")
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold = 0.90

print("Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Camera not detected.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX

print("Loading model...")
model = load_model('MyTrainingModel.h5')
print("Model loaded successfully.")

def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def get_className(classNo):
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        if probabilityValue > threshold:
            color = (0, 255, 0) if classIndex == 0 else (50, 50, 255)
            cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(imgOriginal, (x, y-40), (x+w, y), color, -2)
            cv2.putText(imgOriginal, get_className(classIndex), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
