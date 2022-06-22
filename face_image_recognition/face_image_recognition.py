
import cv2
import face_recognition
import dlib


# Load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file("../images/samples/modi.jpg")
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file("../images/samples/trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

# Save the encodings and the corresponding lables in separate arrays in the same order
known_face_encodings = [modi_face_encoding,trump_face_encoding]
known_face_names = ["Narendra Modi","Donald Trump"]

# load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file("../images/testing/trump-modi.jpg")
# Read Image
image_to_detect = cv2.imread("../images/testing/trump-modi.jpg")

all_face_locations = face_recognition.face_locations(image_to_recognize, model="hog")
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)
print("There are {} faces in this image".format(len(all_face_locations)))


# looping through the face locations  and the face embeddings
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face at location top: {},right:{},bottom:{},left:{}, right: {}'.format(top_pos,right_pos,bottom_pos,left_pos,right_pos))

    # see if the face is any match(es) for the known face(s)
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    # Initialize a name string as unknown face
    name_of_person = "Unknown Face"

    # If a match was found in known_face_encodings,use the first one
    # check if the all_matches have atleast one item
    # if yes, get the index number of the face that is located in the first index of all_matches
    # get the name of corresponding to the index number and save it in the
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]

    # Draw blue rectangle around the face
    # Args: 1. Image 2. position 3 Position 4. BGR color 5. Thickness
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)

    # write name below face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect,name_of_person,(left_pos,bottom_pos),font,0.5,(255,255,255),1)

cv2.imshow("Identified Face ",image_to_detect)
cv2.waitKey(0)


