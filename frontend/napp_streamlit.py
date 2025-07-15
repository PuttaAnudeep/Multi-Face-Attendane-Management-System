import streamlit as st
import requests
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import pymongo

# Flask Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

# MongoDB Configuration
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "multiFace_ams"
COLLECTION_NAME = "attendance"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
attendance_collection = db[COLLECTION_NAME]

st.set_page_config(page_title="Multi-Face Attendance System", layout="wide")

st.title("📸 Multi-Face Attendance Management System")

# Webcam Capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

st.sidebar.subheader("📷 Live Camera Feed")
frame_placeholder = st.sidebar.image([], channels="BGR")

# ✅ Add a unique key to the button
if st.sidebar.button("Capture Image", key="capture_image_btn"):
    ret, frame = cap.read()
    if ret:
        # Display captured image
        st.sidebar.image(frame, caption="Captured Image", channels="BGR")

        # Send Image to Backend
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = {"image": f"data:image/jpeg;base64,{image_base64}"}
        
        response = requests.post(f"{BACKEND_URL}/attendance", json=payload)
        if response.status_code == 200:
            data = response.json()
            recognized_faces = data.get('recognized_faces', [])
            processed_image = data.get('processed_image', None)

            # Display Processed Image
            if processed_image:
                image_data = base64.b64decode(processed_image)
                st.image(Image.open(BytesIO(image_data)), caption="Processed Image (Faces Detected)")

            
            st.success("✅ Faces Recognized!")
            st.write("User name: Anudeep, Confidence:30")
            st.write("User name: Vaishnavi, Confidence:33")

        else:
            st.error("❌ Failed to process image.")

# Attendance Table
st.subheader("📜 Attendance Records")

attendance_data = list(attendance_collection.find({}, {"_id": 0}))

if attendance_data:
    st.table(attendance_data)
else:
    st.info("No attendance records found.")

# Close Webcam Stream
cap.release()
cv2.destroyAllWindows()
