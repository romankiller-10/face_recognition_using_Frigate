# MQTT Constants  
BROKER_HOST = "192.168.0.63"  
BROKER_PORT = 1883  
MQTT_USERNAME = "admin"  
MQTT_PASSWORD = "1BeachHouse@2023"  
MQTT_TOPIC = "frigate/events"  

# Paths  
STORAGE_DIR = "/home/admin/storage"
FRIGATE_SERVER_ADDRESS = "http://jupyterpi:5000"
FRIGATE_DB_PATH = "/home/admin/config/frigate.db"  
EVENTS_DB_PATH = "/home/admin/config/events.db"  
CLIPS_PATH = "/home/admin/storage/clips/"  
RECORDINGS_PATH = "/home/admin/storage/recordings/"  
MODEL_WEIGHTS_PATH = "model/facenet_keras_weights.h5"  
ENCODINGS_PATH = 'encodings/data_augmented_encodings.pkl'  
CAMERA_NAME = 'GarageCamera'
# Face Recognition  
CONFIDENCE_THRESHOLD = 0.8  
RECOGNITION_THRESHOLD = 0.5  
REQUIRED_IMAGE_SIZE = (160, 160)  

# Database Schema  
EVENT_TABLE_SCHEMA = """  
    CREATE TABLE IF NOT EXISTS event (  
        id TEXT PRIMARY KEY,   
        label TEXT,   
        camera TEXT,   
        start_time DATETIME,  
        end_time DATETIME,  
        thumbnail TEXT,  
        sub_label TEXT,  
        snapshot_path TEXT,  
        video_path TEXT  
    )  
"""