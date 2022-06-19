import cv2
import face_recognition
import dlib
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

# face expression model initialization
face_exp_model = model_from_json(open("../datasets/facial_expression_model_structure.json", "r").read())

# load weights into model
face_exp_model.load_weights("../datasets/facial_expression_model_weights.h5")

# list of emotions labels
emotions_label = ('angry','fear','happy','sad','surprise','neutral','1','2','3','4')


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

        # ***************Face detection Starts***************
        # printing the location of current face
        print('Found face {} at top: {},right:{},bottom:{},left:{}'.format(index + 1, top_pos, right_pos, bottom_pos,
                                                                           left_pos))
        # Slicing the current face from current main page
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # Draw rectangle around face
        # Argument: 1. left & top position2. right & bottom 3. color: BGR  4. thickness of the border
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        # ***************Face detection ends***************

        # ***************Emotion detection Starts***************
        # preprocess input, convert it to am image like as the data in dataset
        # convert to grayscale
        current_face_image = cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
        # resize to 48*48 px size
        current_face_image = cv2.resize(current_face_image,(48,48))

        # convert the PIL image into a 3d numpy array
        img_pixels = image.img_to_array(current_face_image)

        # expand the shape of an array into single row multiple columns
        # axis =0 means single dimension
        img_pixels = np.expand_dims(img_pixels,axis=0)

        # pixels are in range of [0,255].normalize all pixels in scale of [0,1]
        # 0 : completely black and 255 : completely black
        img_pixels /= 255
        # ***************Emotion detection Ends***************

        # ***************Emotion Prediction Starts***************
        # do prediction using model,get the prediction values for all 7 expression
        exp_predictions = face_exp_model.predict(img_pixels)

        # find max indexed prediction value(0 till 7)
        max_index = np.argmax(exp_predictions[0])
        print(f"max_index: {max_index}")
        # get corresponding label from emotions label
        emotion_label = emotions_label[max_index]

        # display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotion_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        # ***************Emotion Prediction Ends***************

    # showing the current face with rectangle drawn
    cv2.imshow("Webcam Video ", current_frame)





        # Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam resources
   # webcam_video_stream.read()
    #cv2.destroyAllWindows()
