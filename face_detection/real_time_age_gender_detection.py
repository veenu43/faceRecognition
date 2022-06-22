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
        left_pos = left_pos * 4
        # printing the location of current face
        #print('Found face {} at top: {},right:{},bottom:{},left:{}'.format(index + 1, top_pos, right_pos, bottom_pos,left_pos))
        # Slicing the current face from current main page
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated by using the numpy.mean()
        # mean values for channels,height and width
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)

        # create blob of current flace slice
        # Arguments:   1. current face image 2. scale factor (1 means 100%) 3. Size of the blob 227 bytes 4. mean value 5. swap Red or Blue
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image,1,(227,227),AGE_GENDER_MODEL_MEAN_VALUES,swapRB=False)

        # ***************Gender Prediction Starts***************
        gender_label_list = ['Male','Female']
        gender_protext = "../datasets/gender_deploy.prototxt"
        gender_caffemodel = "../datasets/gender_net.caffemodel"

        # Create Model from files and provide blob as input
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel,gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]
        # ***************Gender Prediction ends***************

        # ***************Gender Prediction Starts***************
        age_label_list = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
        age_protext = "../datasets/age_deploy.prototxt"
        age_caffemodel = "../datasets/age_net.caffemodel"
        # Create Model from files and provide blob as input
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(current_face_image_blob)
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]
        # ***************Gender Prediction ends***************


        # Draw rectangle around face
        # Argument: 1. left & top position2. right & bottom 3. color: BGR  4. thickness of the border
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,gender+" "+age+" "+"yrs",(left_pos,bottom_pos+20),font,0.5,(0,255,0),1)
    # showing the current face with rectangle drawn
    cv2.imshow("Webcam Video ", current_frame)





        # Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam resources
   # webcam_video_stream.read()
    #cv2.destroyAllWindows()
