import cv2
import numpy as np

from FacialRecognitionHandler import prepare_training_data
from FacialRecognitionHandler import predict
#from FacialRecognitionHandler import add_image_to_database

#no label 0 in our training data so person name for label 0 is empty
subjects = ["", "Alec Ngai", "Kendall Jenner", "Hillary Ngai"]

print("Preparing training data...")
faces, labels = prepare_training_data("C:\\Users\\Hilla\\PycharmProjects\\FacialRecognition\\TrainingData")
print("Data is prepared.")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create(threshold = 70)
face_recognizer.train(faces, np.array(labels))

print("Recognizing faces...")
test_path1 = 'C:\\Users\\Hilla\\PycharmProjects\\FacialRecognition\\TestData\\test2.jpg'
test_path2 = 'C:\\Users\\Hilla\\PycharmProjects\\FacialRecognition\\TestData\\test8.jpg'
test_img1 = cv2.imread(test_path1)
test_img2 = cv2.imread(test_path2)

# perform a prediction
predicted_img1, label1 = predict(test_img1, subjects, face_recognizer)
predicted_img2, label2 = predict(test_img2, subjects, face_recognizer)
print("Recognized!")

# display both images
cv2.imshow(label1, cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(label2, cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
is_correct_prediction = raw_input("Was the prediction correct? Enter \"yes\" or \"no\": ")

if is_correct_prediction.lower() == "yes":
    print("Adding image to database...")
    add_image_to_database(test_path1, subjects, label1)

elif is_correct_prediction.lower() == "no":
    label = raw_input("Enter actual subject name: ")
    print("Adding image to database...")
    add_image_to_database(test_path1, subjects, label)

else:
    print("Could not process answer. Please enter \"yes\" or \"no\".")
"""





