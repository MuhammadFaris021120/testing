import cv2  # OpenCV for face detection
from PIL import Image
import glob
import os
import numpy as np  # Needed for OpenCV image conversions

def DetectFace(image, faceCascade, returnImage=False):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV's detectMultiScale
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))

    # If faces are found and returnImage is True, draw rectangles
    if returnImage:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image
    else:
        return faces

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]
    xDelta = max(cropBox[2] * (boxScale - 1), 0)
    yDelta = max(cropBox[3] * (boxScale - 1), 0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box = [cropBox[0] - xDelta, cropBox[1] - yDelta, cropBox[0] + cropBox[2] + xDelta, cropBox[1] + cropBox[3] + yDelta]

    return image.crop(PIL_box)

def process_images(input_folder, output_folder, boxScale=1):
    # Load Haarcascade model for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.jpeg"))

    if not image_paths:
        print("No images found in the input folder.")
        return

    for imagePath in image_paths:
        pil_im = Image.open(imagePath)
        cv_im = np.array(pil_im)

        faces = DetectFace(cv_im, faceCascade)

        if faces is not None and len(faces) > 0:
            fname = os.path.basename(imagePath)  # Extract filename (e.g., "image1.png")
            name, ext = os.path.splitext(fname)

            n = 1
            for face in faces:
                croppedImage = imgCrop(pil_im, face, boxScale=boxScale)

                # Save cropped image in the output folder
                croppedImage.save(os.path.join(output_folder, f"{name}_crop{n}{ext}"))
                n += 1

            print(f"Processed {fname} - {len(faces)} faces detected.")
        else:
            print(f"No faces found in {imagePath}.")

# Example Usage
input_folder = r"C:\Work\faceCrop\Before"
output_folder = r"C:\Work\faceCrop\after"
process_images(input_folder, output_folder, boxScale=1.5)
