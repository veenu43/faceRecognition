
import cv2
import face_recognition

image = face_recognition.load_image_file("20190715_123912.jpg")
face_locations = face_recognition.face_locations(image)

image_to_detect = cv2.imread("20190715_123912.jpg")
cv2.imshow("test",image_to_detect)

all_face_locations = face_recognition.face_locations(image_to_detect,model="hog")
print("There are {} faces in this image".format(len(all_face_locations)))