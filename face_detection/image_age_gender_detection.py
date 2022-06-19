
import cv2
import face_recognition
import dlib

# Read Image
image_to_detect = cv2.imread("../images/testing/trump-modi.jpg")
#cv2.imshow("test",image_to_detect)

# Hog model
# find total faces in image
all_face_locations = face_recognition.face_locations(image_to_detect,number_of_times_to_upsample=3,model="hog")
print("There are {} faces in this image".format(len(all_face_locations)))

# CNN Model
#all_face_locations = face_recognition.face_locations(image_to_detect,number_of_times_to_upsample=1,model="cnn")
#print("There are {} faces in this image".format(len(all_face_locations)))


# Extract faces from image
for index,current_face_location in enumerate(all_face_locations):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Found face {} at top: {},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))

    # Slice image to get faces
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    # The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated by using the numpy.mean()
    # mean values for channels,height and width
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # create blob of current flace slice
    # Arguments:   1. current face image 2. scale factor (1 means 100%) 3. Size of the blob 227 bytes 4. mean value 5. swap Red or Blue
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES,
                                                    swapRB=False)

    # ***************Gender Prediction Starts***************
    gender_label_list = ['Male', 'Female']
    gender_protext = "../datasets/gender_deploy.prototxt"
    gender_caffemodel = "../datasets/gender_net.caffemodel"

    # Create Model from files and provide blob as input
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    gender_cov_net.setInput(current_face_image_blob)
    gender_predictions = gender_cov_net.forward()
    gender = gender_label_list[gender_predictions[0].argmax()]
    # ***************Gender Prediction ends***************

    # ***************Gender Prediction Starts***************
    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
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
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender + " " + age + " " + "yrs", (left_pos, bottom_pos + 20), font, 0.5, (0, 255, 0), 1)
    # showing the current face with rectangle drawn
cv2.imshow("Image Age Gender Detection ", image_to_detect)
cv2.waitKey(0)



