import cv2, time
import mediapipe as mp
import numpy as np


mp_objectron = mp.solutions.objectron
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(2)

iterations = [1, 2, 5, 6]
h = np.load('h.npy')
temp_img = cv2.imread('template.jpg')
while True:
    _, img = vid.read()
    with mp_objectron.Objectron(static_image_mode=True, max_num_objects=6, min_detection_confidence=0.5, model_name='Shoe') as objectron:
   
      results = objectron.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      if not results.detected_objects:
        print('No box landmarks detected on shoe')
        continue
      print('Box landmarks of shoe:')
      ann_img = img.copy()
      for detected_object in results.detected_objects:
        for i in iterations:
          x = int(detected_object.landmarks_2d.landmark[i].x * img.shape[1])
          y = int(detected_object.landmarks_2d.landmark[i].y * img.shape[0])
          ann_img = cv2.circle(ann_img, (x,y), 1, color=(0, 0, 255), thickness=5)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      warp_img = cv2.warpPerspective(ann_img, h, (img.shape[1],img.shape[0]))
      cv2.imshow('img', warp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()