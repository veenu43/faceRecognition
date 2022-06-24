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
    # convert the pil image to draw object with Alpha Mode for Translucency
    d = ImageDraw.Draw(pil_image,"RGBA")

    # Make left, right eyebrows darker
    # Polygon on top and line on bottom with dark color
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Add lipstick to top and bottom lips
    # using red polygons and lines filled with red
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 100))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 100))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # Make left and right eyes filled with red
    d.polygon(face_landmarks['left_eye'], fill=(255, 0, 0, 100))
    d.polygon(face_landmarks['right_eye'], fill=(255, 0, 0, 100))
    d.line(face_landmarks['left_eye'], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'], fill=(0, 0, 0, 110), width=6)

    # Display Image
    pil_image.show()

    # you can also save a copy of the new image to disk
    # pil_image.save("vinit_makeup.jpg")
