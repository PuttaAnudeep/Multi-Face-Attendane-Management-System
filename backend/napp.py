from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime
from pymongo import MongoClient
import base64
import json
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

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
embeddings_collection = db['embeddings']

# Paths
DATASET_PATH = "E:/computer technology/Summer_Internship 2024(Class Attendance Management system)/face_img"

# Face Detector & Recognizer
detector = MTCNN()
embedder = FaceNet()

# Mapping User IDs to Names
label_to_name = {
    "user1": "Anudeep",
    "user2": "Akhil",
    "user3": "Sreeyas"
}

stored_embeddings = {}

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
            mark_attendance_mongodb(user['user_id'], user['name'])

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
    print(f"✅ Attendance marked for {name} in MongoDB")

def recognize_faces(image_path):
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("❌ Error: Unable to load image. Check the file path.")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # FaceNet expects RGB
        original_image = image.copy()

        # Detect faces using MTCNN
        detections = detector.detect_faces(rgb_image)
        print(f"🔍 Detected Faces: {len(detections)}")

        recognized_faces_data = []

        for detection in detections:
            x, y, w, h = detection['box']
            x, y = max(0, x), max(0, y)  # Ensure values are within image bounds

            # Extract and resize face region for FaceNet
            face_region = rgb_image[y:y + h, x:x + w]
            face_resized = cv2.resize(face_region, (160, 160))
            face_resized = np.expand_dims(face_resized, axis=0)  # Expand dims for FaceNet

            # Generate face embedding
            face_embedding = embedder.embeddings(face_resized)[0]

            # Compare with stored embeddings using cosine similarity
            best_match = None
            best_score = 1.0  # Higher score = less similar

            for user_id, stored_embedding in stored_embeddings.items():
                score = cosine(face_embedding, stored_embedding)
                if score < best_score:  # Lower score = better match
                    best_match = user_id
                best_score = score

            # Confidence threshold
            if best_score < 0.6:
                name = label_to_name.get(best_match, "Unknown")
                confidence = round((1 - best_score) * 100, 2)  # Convert similarity to confidence

                recognized_faces_data.append({
                    'user_id': best_match,
                    'name': name,
                    'confidence': confidence,
                    'x': x, 'y': y, 'w': w, 'h': h
                })
                print(f"✅ Recognized: {name} (ID: {best_match}, Confidence: {confidence:.2f}%)")
            else:
                print(f"❌ Unrecognized face (Similarity: {1 - best_score:.2f})")

            # Draw bounding box
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert image to base64 for return
        _, buffer = cv2.imencode('.jpg', original_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return recognized_faces_data, img_base64
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return [], None  # Return empty results in case of failure
# Train Face Embeddings (Instead of LBPH)
def train_face_embeddings():
    embeddings = {}
    for user_folder in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user_folder)
        if os.path.isdir(user_path):
            for file in os.listdir(user_path):
                if file.endswith(("jpg", "png")):
                    image_path = os.path.join(user_path, file)
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb_image)

                    for face in faces:
                        x, y, w, h = face['box']
                        face_region = rgb_image[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_region, (160, 160))

                        # Extract embeddings
                        face_embedding = embedder.embeddings([face_resized])[0]
                        embeddings[user_folder] = face_embedding.tolist()

    # Store embeddings in MongoDB
    embeddings_collection.insert_one({"data": embeddings})
    print("✅ Face embeddings trained and stored in MongoDB")

# Load Embeddings at Startup
def load_embeddings():
    global stored_embeddings
    stored_embeddings = embeddings_collection.find_one({}, {"_id": 0, "data": 1})["data"]
    print("✅ Loaded embeddings from MongoDB")

if __name__ == '__main__':
    if embeddings_collection.count_documents({}) == 0:
        train_face_embeddings()
    load_embeddings()
    app.run(debug=True)
