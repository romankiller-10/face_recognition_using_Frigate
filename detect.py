import cv2   
import numpy as np  
import mtcnn  
from architecture import *  
from train import normalize, l2_normalizer  
from scipy.spatial.distance import cosine  
import pickle  
import os  
import time   

# Import constants  
from constants import (MODEL_WEIGHTS_PATH, ENCODINGS_PATH, REQUIRED_IMAGE_SIZE, CONFIDENCE_THRESHOLD, RECOGNITION_THRESHOLD, CLIPS_PATH, RECORDINGS_PATH, CAMERA_NAME)  

# Initialize constants and models  
face_encoder = InceptionResNetV2()  
face_encoder.load_weights(MODEL_WEIGHTS_PATH)  
face_detector = mtcnn.MTCNN()  
with open(ENCODINGS_PATH, 'rb') as f:  
    encoding_dict = pickle.load(f)  

def get_face(img, box):  
    x1, y1, width, height = box  
    x1, y1 = abs(x1), abs(y1)  
    x2, y2 = x1 + width, y1 + height  
    face = img[y1:y2, x1:x2]  
    return face, (x1, y1), (x2, y2)  

def get_encode(face_encoder, face, size):  
    face = normalize(face)  
    face = cv2.resize(face, size)  
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]  
    return encode  

def load_pickle(path):  
    with open(path, 'rb') as f:  
        encoding_dict = pickle.load(f)  
    return encoding_dict  

def detect(img, box, detector, encoder, encoding_dict):  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    results = detector.detect_faces(img_rgb)  
    print(results)  
    name = ''  
    for res in results:  
        print(f"confidence:{res['confidence']}")  
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])  
        encode = get_encode(encoder, face, REQUIRED_IMAGE_SIZE)  
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]  
        
        distance = float("inf")  
        detected_name = "Unknown"  
        for db_name, db_encode in encoding_dict.items():  
            dist = cosine(db_encode, encode)  
            print(f"{db_name}: {dist}")  
            if dist < RECOGNITION_THRESHOLD and dist < distance:  
                detected_name = db_name  
                distance = dist  
        if detected_name == "Unknown":  
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)  
            cv2.putText(img, 'Unknown', pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)  
        else:  
            name += detected_name + '  '  
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)  
            cv2.putText(img, detected_name + f'__{distance:.2f}', (pt_1[0], pt_1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1,  
                        (0, 200, 200), 2)  
    if name == '':  
        return img, 'Unknown person'  
    return img, name  


def generate_recognized_image(event_data, date_format):  

    event_id = event_data['id']  
    box = event_data['snapshot']['box']  
    source_img_path = os.path.join(CLIPS_PATH, f"{CAMERA_NAME}-{event_id}-bestinsec.png")  
    out_image_path = os.path.join(CLIPS_PATH, f"{CAMERA_NAME}-{event_id}-person-rec.jpg")  

    folder_path = os.path.join(RECORDINGS_PATH, date_format[:10], '00')  
    creation_hour = get_folder_creation_hour(folder_path)  
    if isinstance(creation_hour, str):  
        print(creation_hour)  # Log the error message  
        creation_hour = 0  # Default to 0 if folder does not exist  

    subfolder_num = int(date_format[11:13]) - int(creation_hour)  
    subfolder_num_str = str(subfolder_num).zfill(2)  
    video_path = os.path.join(RECORDINGS_PATH, date_format[:10], subfolder_num_str, CAMERA_NAME, f"{date_format[14:16]}.{date_format[17:19]}.mp4")  
    img = cv2.imread(source_img_path)  

    if img is None:  
        print("Image not loaded. Check the path.")  
        return "unknown", None, None  
    else:  
        out_img, name = detect(img, box, face_detector, face_encoder, encoding_dict)  
        cv2.imwrite(out_image_path, out_img)  
        if name:  
            return name, out_image_path, video_path  
        else:  
            return "unknown", out_image_path, video_path  


def get_folder_creation_hour(folder_path):  
    try:  
        creation_time = os.path.getctime(folder_path)  
        creation_time_struct = time.localtime(creation_time)  
        creation_hour = creation_time_struct.tm_hour  
        return creation_hour  
    except FileNotFoundError:  
        return "The specified folder does not exist."