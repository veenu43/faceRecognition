
import cv2
import face_recognition
import dlib


# Load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file("../images/samples/modi.jpg")
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file("../images/samples/trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

vinit_image = face_recognition.load_image_file("../images/samples/Vinit.jpg")
vinit_face_encoding = face_recognition.face_encodings(vinit_image)[0]

# Save the encodings and the corresponding lables in separate arrays in the same order
known_face_encodings = [modi_face_encoding,trump_face_encoding,vinit_face_encoding]
known_face_names = ["Narendra Modi","Donald Trump","Vinit"]

# load the unknown image to recognize faces in it
image_to_recognize_path = "../images/testing/trump.jpg"
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
# Read Image
image_to_detect = cv2.imread(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]

# find the distance of current encoding with all known encodings
face_distances = face_recognition.face_distance(known_face_encodings,image_to_recognize_encodings)

# print face distance for each known sample to the unknown image
for i,face_distance in enumerate(face_distances):
    print("The calculated face distance is {:.2} from sample image {}".format(face_distance,known_face_names[i]))
