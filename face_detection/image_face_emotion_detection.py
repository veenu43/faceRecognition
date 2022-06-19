
import cv2
import face_recognition
import dlib
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

print(f"cv2 version {cv2.__version__}")
print(f"face_recognition version {face_recognition.__version__}")
print(f"dlib version {dlib.__version__}")
image1 = face_recognition.load_image_file("../images/20190715_123912.jpg")
face_locations = face_recognition.face_locations(image1)

# Read Image
current_frame = cv2.imread("../images/manypeoples.jpg")
#cv2.imshow("test",image_to_detect)

# Hog model
# find total faces in image
all_face_locations = face_recognition.face_locations(current_frame,number_of_times_to_upsample=3,model="hog")
print("There are {} faces in this image".format(len(all_face_locations)))



# face expression model initialization
face_exp_model = model_from_json(open("../datasets/facial_expression_model_structure.json","r").read())

# load weights into model
face_exp_model.load_weights("../datasets/facial_expression_model_weights.h5")

# list of emotions labels
emotions_label = ('angry','disgust','fear','happy','sad','surprise','neutral')


# Extract faces from image
for index,current_face_location in enumerate(all_face_locations):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top: {},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))

    # Slice image to get faces
    current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
    # Draw rectangle around face
    # Argument: 1. left & top position2. right & bottom 3. color: BGR  4. thickness of the border
    cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    # ***************Face detection ends***************

    # ***************Emotion detection Starts***************
    # preprocess input, convert it to am image like as the data in dataset
    # convert to grayscale
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
    # resize to 48*48 px size
    current_face_image = cv2.resize(current_face_image, (48, 48))

    # convert the PIL image into a 3d numpy array
    img_pixels = image.img_to_array(current_face_image)

    # expand the shape of an array into single row multiple columns
    # axis =0 means single dimension
    img_pixels = np.expand_dims(img_pixels, axis=0)

    # pixels are in range of [0,255].normalize all pixels in scale of [0,1]
    # 0 : completely black and 255 : completely black
    img_pixels /= 255
    # ***************Emotion detection Ends***************

    # ***************Emotion Prediction Starts***************
    # do prediction using model,get the prediction values for all 7 expression
    exp_predictions = face_exp_model.predict(img_pixels)

    # find max indexed prediction value(0 till 7)
    max_index = np.argmax(exp_predictions[0])
    # get corresponding label from emotions label
    emotion_label = emotions_label[max_index]

    # display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    # ***************Emotion Prediction Ends***************

# showing the current face with rectangle drawn
cv2.imshow("Image Face Emotions ", current_frame)
cv2.waitKey(0)



