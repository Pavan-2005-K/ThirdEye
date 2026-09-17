import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def detect_faces(image_path: str):
    """
    Detect faces in an image.

    Returns:
        List of detected face bounding boxes.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    return [list(face) for face in faces]