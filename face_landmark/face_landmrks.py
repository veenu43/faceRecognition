import face_recognition
from PIL import Image, ImageDraw

# Load the jpg file into a numpy array
face_image = face_recognition.load_image_file("../images/samples/Vinit.jpg")

# Find all facial landmarks in all the faces  in the image
face_landmarks_list = face_recognition.face_landmarks(face_image)

# print all the face landmarks
print(face_landmarks_list)

# for loop to iterate through all lan marks in the list
for face_landmarks in face_landmarks_list:
    #print("Landmarks: ",face_landmarks)

    # convert numpy array to pil image Object
    pil_image = Image.fromarray(face_image)
    # convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)

    d.line(face_landmarks['chin'],fill=(255,255,255),width=2)
    d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
    d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)

    # Display Image
    pil_image.show()
