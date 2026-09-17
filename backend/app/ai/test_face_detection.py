import os

from app.ai.face_detection import detect_faces


image_folder = "uploads/sketches"

files = os.listdir(image_folder)

if not files:
    print("No sketch images found.")
    exit()

image_path = os.path.join(
    image_folder,
    files[0]
)

print("Testing image:")
print(image_path)

faces = detect_faces(image_path)

print("Number of faces detected:", len(faces))

if faces:
    print("Detected face coordinates:")
    print(faces)
else:
    print("No face detected.")