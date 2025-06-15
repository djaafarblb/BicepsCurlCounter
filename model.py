import cv2
import PIL
import numpy as np
from sklearn.svm import  LinearSVC

class Model:
    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = []  # Use a Python list instead of np.array initially
        class_list = []

        for i in range(1, counters[0]):
            img = cv2.imread(f"1/frame{i}.jpg")[:, :, 0]
            img = img.reshape(16950)
            img_list.append(img)  # Append directly to the list
            class_list.append(1)  # Append to class_list as well

        for i in range(1, counters[1]):
            img = cv2.imread(f"2/frame{i}.jpg")[:, :, 0]
            img = img.reshape(16950)
            img_list.append(img)
            class_list.append(2)

        # Convert the list to a NumPy array
        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        print('Model Successfly trained')

    def  predict(self, frame):
        frame = frame[1]
        cv2.imwrite('frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = PIL.Image.open('frame.jpg')
        img.thumbnail((150,150), PIL.Image.Resampling.LANCZOS)
        img.save('frame.jpg')

        img = cv2.imread('frame.jpg')[:,:,0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return  prediction[0]


