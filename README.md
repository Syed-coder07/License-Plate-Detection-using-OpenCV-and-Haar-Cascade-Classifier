## License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier

## Aim:

To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used:

* Python 3.7 or above
* OpenCV (opencv-python)
* NumPy
* Matplotlib
* Jupyter Notebook (Anaconda)
* Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm:

* Import necessary libraries such as OpenCV and Matplotlib
* Read the input vehicle image
* Convert the original image to grayscale for faster computation
* Load the Haar Cascade classifier for license plate detection
* Detect license plate using detectMultiScale function
* Draw rectangle around detected area
* Crop the detected region using numpy slicing with (x, y, w, h) values
* Apply median blurring on the cropped region
* Replace the original region with blurred version
* Display final result using Matplotlib

## Program:

## Developed by: Syed Shamsheer Ali B

## Register Number: 212225220114

```
import cv2
import matplotlib.pyplot as plt
import urllib.request
xml_path = r"C:\Users\syedf\OneDrive\Documents\DIPT\haarcascade_russian_plate_number.xml"
url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_russian_plate_number.xml"
urllib.request.urlretrieve(url, xml_path)
img = cv2.imread(r"C:\Users\syedf\OneDrive\Documents\DIPT\car_plate.jpg")
plate_cascade = cv2.CascadeClassifier(xml_path)
print("Image loaded:", img is not None)
print("Cascade loaded:", not plate_cascade.empty())
if img is not None and not plate_cascade.empty():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4
    ) 
    result = img.copy()
    for (x, y, w, h) in plates:
        roi = result[y:y+h, x:x+w]
        result[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51, 51), 0)
    print("Total number plates detected:", len(plates))
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Number Plate Detection")
    plt.axis("off")
    plt.show()
```

## Output:

<img width="971" height="577" alt="image" src="https://github.com/user-attachments/assets/2af73975-cc05-4826-8fc3-bf9ef5782639" />

## Modification Done:

Parameter tuning was performed by adjusting scaleFactor and minNeighbors values in detectMultiScale to improve accuracy and reduce false detections. Median blur was applied to protect license plate information.

## Result:

The License Plate Detection system was successfully implemented using OpenCV and Haar Cascade. The detected license plate region was blurred using median filtering. The modified values improved overall detection performance and output quality.

## Conclusion:

This workshop demonstrates how classical computer vision methods like Haar Cascades can be used for real-time applications such as automated toll systems, smart parking, and traffic surveillance. Proper preprocessing and parameter tuning significantly improve detection results.
