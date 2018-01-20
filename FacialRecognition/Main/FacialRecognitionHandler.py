import cv2
import os
import shutil

from pathlib import Path
from LabelPhotos import draw_rectangle
from LabelPhotos import draw_text

def detect_face(img):
    #convert trial images to gray since openCV face detector uses gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('C:\Users\Hilla\PycharmProjects\FacialRecognition\OpenCVFiles\haarcascade_frontalface_default.xml')

    #detect multi-scale outputs a list of faces, scaling factor specifies how much you minimize the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return null
    if 0 == len(faces):
        return None, None

    #assuming the image only has one face, take the area of the first one
    (x, y, w, h) = faces[0]

    #return only the face area of the gray image as well as the face area coordinates
    return gray[y:y + w, x:x + h], faces[0]



#function will read each person's training images, detect the face,
#and return a list of faces and a corresponding list of labels
def prepare_training_data(data_folder_path):

    #get directories for each person
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    #go through each directory and read images within the directory
    for dir_name in dirs:

        #ignore any directories with a different naming convention
        if not dir_name.startswith("P"):
            continue;

        #labels are just positive integers from the name of the directory
        label = int(dir_name.replace("P", ""))

        #get the name of the person's directory
        # ex: "training-data/s1"
        subject_dir_path = data_folder_path + "\\" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        #for each image, detect face, and add to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "\\" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(18)

            face, rect = detect_face(image)

            #ignore any faces that aren't detected
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels



#recognizes person in the image and labels it
def predict(test_img, subjects, face_recognizer):

    #don't want to change original image
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)

    if label == -1:
        label_text = "Unable to recognize face."

    else:
        label_text = subjects[label] + " - Confidence Interval: " + str(confidence)

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 6)

    return img, subjects[label]



def subject_exists_in_database(name, subjects):
    if name.title() in subjects:
        return True;



def image_exists_in_database(test_image_path, subject_name, subject):
    image_name = os.path.basename(test_image_path)
    training_file = Path("C:\\Users\\Hilla\\PycharmProjects\\FacialRecognition\\TrainingData\\" + image_name)

    if training_file.is_file():
        return True;
    else:
        return False;


#finish this
def add_image_to_data_base(test_image_path, subjects, subject_name):
    if subject_exists_in_database():
        label = subjects.index(subject_name.title())
        training_image_path = "C:\\Users\\Hilla\\PycharmProjects\\FacialRecognition\\TrainingData\\P" + label
        shutil.copy(test_image_path, training_image_path)

