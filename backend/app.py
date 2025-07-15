from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import base64
import json
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Configuration
MONGO_URI = "mongodb://127.0.0.1:27017/"  # Ensure MongoDB is running
DB_NAME = "multiFace_ams"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
attendance_collection = db['attendance']
# Paths
DATASET_PATH = "E:/computer technology/Summer_Internship 2024(Class Attendance Management system)/face_img"
MODEL_PATH = "trained_model.yml"
LABELS_PATH = "labels.json"
# Load Face Recognizer Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Create face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8)
# Map User IDs to Names
label_to_name = {
    "user1": "Anudeep",
    "user2": "Akhil",
    "user3": "Sreeyas"
}
if os.path.exists("trained_model.yml"):
    recognizer.read("trained_model.yml")

@app.route('/')
def hello_world():
    return 'Hello, World! API is running.'

@app.route('/attendance', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        decoded_data = base64.b64decode(image_data)

        # Convert image to OpenCV format
        nparr = np.frombuffer(decoded_data, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        cv2.imwrite(filename, image_np)

        recognized_faces, processed_img_base64 = recognize_faces(filename)

        # Mark attendance for recognized faces
        for user in recognized_faces:
            mark_attendance_mongodb(user['user_id'])

        return jsonify({'recognized_faces': recognized_faces, 'processed_image': processed_img_base64})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/names', methods=['POST'])
def receive_names():
    data = request.get_json()
    named_faces = data['names']

    for face in named_faces:
        mark_attendance_mongodb(face['user_id'], face['name'])

    return jsonify({'message': 'Names received successfully'})

# Function to store attendance data in MongoDB
def mark_attendance_mongodb(user_id, name="Unknown"):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    if attendance_collection.find_one({'id': user_id, 'date': date}):
        print(f"{name} is already marked present for today.")
        return

    attendance_data = {
        'id': user_id,
        'name': name,
        'date': date,
        'time': time
    }
    attendance_collection.insert_one(attendance_data)
    print(f"Attendance marked for {name} in MongoDB")

def recognize_faces(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    print("🔍 Detected Faces:", len(faces))
    
    recognized_faces_data = []

    # Load trained recognizer model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trained_model.yml")

    # Map user IDs to names
    user_mapping = {1: "Anudeep", 2: "Akhil", 3: "Sreeyas"}

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (100, 100))

        user_id, confidence = recognizer.predict(face_resized)

        if confidence < 60:  # Adjust threshold as needed
            name = user_mapping.get(user_id, "Unknown")
            recognized_faces_data.append({
                'user_id': user_id,
                'name': name,
                'confidence': round(confidence, 2),
                'x': x, 'y': y, 'w': w, 'h': h
            })
            print(f"✅ Recognized: {name} (ID: {user_id}, Confidence: {confidence:.2f})")
        else:
            print(f"❌ Unrecognized face (Confidence: {confidence:.2f})")

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', original_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return recognized_faces_data, img_base64

def train_face_recognizer():
    print("🔄 Preparing training data...")

    faces, labels, name_mapping = get_images_and_labels(DATASET_PATH)

    print(f"🎯 Training model with {len(faces)} face samples...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    # Save label-to-name mapping
    with open(LABELS_PATH, "w") as file:
        json.dump(name_mapping, file)

    print(f"✅ Model trained & saved as '{MODEL_PATH}'")
    print(f"📁 Labels saved as '{LABELS_PATH}'")
def get_images_and_labels(dataset_path):
    face_samples = []
    labels = []
    name_mapping = {}  # Stores numeric ID to Name

    label_id = 1  # Start labeling from 1

    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)

        if os.path.isdir(user_path):
            print(f"📂 Processing {user_folder}...")

            # Map 'user1' → 'Anudeep'
            if user_folder in label_to_name:
                name_mapping[label_id] = label_to_name[user_folder]
            else:
                continue  # Skip unknown folders

            for file in os.listdir(user_path):
                if file.endswith(("jpg", "png")):
                    image_path = os.path.join(user_path, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

                    for (x, y, w, h) in faces_detected:
                        face_resized = cv2.resize(image[y:y + h, x:x + w], (100, 100))  # Standardize size
                        face_samples.append(face_resized)
                        labels.append(label_id)  # Assign numerical label

            print(f"✅ {user_folder} → {name_mapping[label_id]} (Label: {label_id})")
            label_id += 1  # Increment for next user
    return face_samples, labels, name_mapping
if __name__ == '__main__':
    dataset_path = "E:/computer technology/Summer_Internship 2024(Class Attendance Management system)/face_img"
    if not os.path.exists("trained_model.yml"):
        train_face_recognizer()
    app.run(debug=True)
