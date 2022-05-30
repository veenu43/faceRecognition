import cv2
import face_recognition
import dlib

# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

# initialize the array variable to hold all face locations in the frame
all_face_locations = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()

    # resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25, fy=0.25)

    # Detect all faces in the image
    # arguments are image, no_of_times_to_upsample,model
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')

    # looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        # splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        # printing the location of current face
        print('Found face {} at top: {},right:{},bottom:{},left:{}'.format(index + 1, top_pos, right_pos, bottom_pos,
                                                                           left_pos))

        # Slice image to get faces
        current_face_image = current_frame_small[top_pos:bottom_pos, left_pos:right_pos]
        cv2.imshow("Face No: " + str(index), current_face_image)
        cv2.waitKey(0)