import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw



# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)
if not webcam_video_stream.isOpened():
    print("Cannot open camera")
    exit()
# initialize the array variable to hold all face locations in the frame
all_face_locations = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()

    # Find all facial landmarks in all the faces  in the image
    face_landmarks_list = face_recognition.face_landmarks(current_frame)

    # print all the face landmarks
    print("no of faces detected: ", len(face_landmarks_list))
    print(face_landmarks_list)

    # convert numpy array to pil image Object
    pil_image = Image.fromarray(current_frame)
    # convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)

    # loop through every face
    index=0
    while index < len(face_landmarks_list):
        # for loop to iterate through all lan marks in the list
        for face_landmarks in face_landmarks_list:
            #print("Landmarks: ",face_landmarks)

            d.line(face_landmarks['chin'],fill=(255,255,255),width=2)
            d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)

        index +=1

    # convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert('RGB')
    rgb_open_cv_image = np.array(rgb_image)

    # Convert RGB to BGR
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image,cv2.COLOR_RGB2BGR)
    rgb_open_cv_image = bgr_open_cv_image[:,:,::-1].copy()

    # showing the current face with rectangle drawn
    cv2.imshow("Webcam Video ", rgb_open_cv_image)

   # Press 'q' on the keyboard to break the while loop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resources
# webcam_video_stream.read()
#cv2.destroyAllWindows()


