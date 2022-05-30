
import cv2
import face_recognition

image = face_recognition.load_image_file("20190715_123912.jpg")
face_locations = face_recognition.face_locations(image)

# Read Image
image_to_detect = cv2.imread("manypeoples.jpg")
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
    cv2.imshow("Face No: "+str(index),current_face_image)




