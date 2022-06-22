import cv2
import face_recognition
import dlib

# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

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

# initialize the array to hold all face locations,encodings and labels in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()

    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25, fy=0.25)

    # Detect all faces in the image
    # arguments are image, no_of_times_to_upsample,model
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')

    # ******* copy from image face recognition ********
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    all_face_names = []

    # looping through the face locations
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        # splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos * 4
        # printing the location of current face
        #print('Found face at location top: {},right:{},bottom:{},left:{}, right: {}'.format(top_pos,right_pos,bottom_pos,left_pos,right_pos))

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
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

        # write name below face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Webcam Video ", current_frame)
    #cv2.waitKey(0)

    # Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam resources
   # webcam_video_stream.read()
    #cv2.destroyAllWindows()
