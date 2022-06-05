#Importing all the important models and install them if not installed on your device
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time




#Starting the camera
cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
