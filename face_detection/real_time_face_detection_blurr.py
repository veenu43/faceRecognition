import cv2
import face_recognition
import dlib

# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)
#webcam_video_stream = cv2.VideoCapture('images/20160424_161800.mp4')
if not webcam_video_stream.isOpened():
    print("Cannot open camera")
    exit()
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
        # Slicing the current face from current main page
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # Blurr the slice face and save it to the same array itself
        # Parameter : 1. Image 2. Tuple : (99,99) - size of kernel(blurred area) 3. 30 : std deviation of the blurr
        current_face_image = cv2.GaussianBlur(current_face_image,ksize=(99,99),sigmaX=30)
        # Paste the blurred face into the actual frame
        current_frame[top_pos:bottom_pos,left_pos:right_pos] = current_face_image

        # Draw rectangle around face
        # Argument: 1. left & top position2. right & bottom 3. color: BGR  4. thickness of the border
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)

    # showing the current face with rectangle drawn
    cv2.imshow("Webcam Video ", current_frame)





        # Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam resources
   # webcam_video_stream.read()
    #cv2.destroyAllWindows()
