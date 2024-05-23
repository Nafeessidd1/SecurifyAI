import face_recognition
import firebase_admin
from firebase_admin import credentials, db
import os
import numpy as np

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-sss-default-rtdb.firebaseio.com/",
    'storageBucket': "gs://face-recognition-sss.appspot.com"
})

# Reference to the database
ref = db.reference('known_faces')

def encode_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    else:
        return None

def upload_face(name, encoding):
    ref.push({
        'name': name,
        'encoding': encoding.tolist()  # Convert numpy array to list for JSON serialization
    })

def main():
    # Directory containing images of known persons
    known_faces_dir = 'known_faces_dir'  # Set this to the correct path

    if not os.path.exists(known_faces_dir):
        print(f"Directory {known_faces_dir} does not exist.")
        return

    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name, _ = os.path.splitext(filename)
            image_path = os.path.join(known_faces_dir, filename)
            encoding = encode_face(image_path)
            if encoding is not None:
                upload_face(name, encoding)
                print(f"Uploaded {name}'s face encoding to Firebase.")
            else:
                print(f"Could not find a face in {filename}")

if __name__ == "__main__":
    main()
