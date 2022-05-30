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

