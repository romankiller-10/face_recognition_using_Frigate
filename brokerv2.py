import paho.mqtt.client as mqtt_client  
import json  
import sqlite3  
import time  
import os  
from datetime import datetime  
import threading  
import logging  
from detect import generate_recognized_image  
from constants import *  

import requests
from io import BytesIO
from PIL import Image

# Initialize global variables  
last_id = None  
date_format = None  
flag = None
# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

# Function called when the client connects to the MQTT broker  
def on_connect(client, userdata, flags, rc):  
    global last_id, date_format  
    last_id = None  
    logging.info("Connected with result code " + str(rc))  
    client.subscribe(MQTT_TOPIC)  

# Function called when a message is received from the MQTT broker  
def on_message(client, userdata, msg):  
    global last_id, date_format, flag  
    payload = msg.payload.decode()  
    try:  
        data = json.loads(payload)  
        event_id = data.get('before', {}).get('id', None)  
        if event_id and ('person' in data.get('before', {}).get('label', None)):  
            if event_id != last_id:  
                last_id = event_id
                flag = time.time()  
                logging.info(f"{datetime.fromtimestamp(data['before']['frame_time'])}: Person detected!")  
                date_format = str(datetime.fromtimestamp(data['before']['frame_time']))  
                logging.info(f"Event_id: {event_id}")  
            elif data["after"]["snapshot"]['frame_time']-data['after']['start_time'] <= 1:  
                logging.info("Event is processing")  

            
            logging.info(flag)
            current_time = time.time()
            if flag == None:
                period = 0
            else:
                period = current_time - flag
            if period > 3:
                flag = None
                logging.info("Since 1 secoonds is passed, trying to get the best img until now ...")
                #Save the best snapshot img
                start_time = time.time()
                snapshot_image = fetch_best_snapshot(event_id)
                file_path = os.path.join(CLIPS_PATH, f"{CAMERA_NAME}-{event_id}-bestinsec.png")
                save_snapshot_image(snapshot_image, file_path)
                current_time = time.time()
                period = current_time - start_time
                logging.info(f"Saved the best snapshot in {period} seconds.")

                thread = threading.Thread(target=process_event, args=(data['after'],))  
                thread.start()  
            
            if data['type'] == 'end' and flag != None:  
                event_length = data['after']['end_time'] - data['after']['start_time']  
                logging.info("Event is finished.(%.1fs)" % event_length)  
                logging.info("Processing snapshots.")  
                flag = None
                thread = threading.Thread(target=process_event, args=(data['after'],))  
                thread.start()  

    except json.JSONDecodeError:  
        logging.error("Payload is not in JSON format")  

# Function to process the event and handle face recognition  
def process_event(event_data):  
    event_id = event_data['id']  
    path = os.path.join(CLIPS_PATH, f"{CAMERA_NAME}-{event_id}-bestinsec.png")  
    if wait_for_file_creation(path):  
        start_time = time.time()  
        recognized_name, out_image_path, video_path = generate_recognized_image(event_data, date_format)  
        logging.info(f"Processing event {event_id} finished in {time.time()-start_time} seconds. Recognized name: {recognized_name}")  

        # Create the payload combining both paths  
        payload = json.dumps({  
            "snapshot_path": out_image_path,  
            "result": f"{recognized_name} was spotted."  
        }) 
        # Publish a message to the new topic  

        result = client.publish(MQTT_TOPIC, payload)  

        # Check if the publish was successful  
        status = result.rc  
        if status == 0:  
            logging.info(f"Sent `{payload}` to topic `{MQTT_TOPIC}`")  
        else:  
            logging.info(f"Failed to send message to topic {MQTT_TOPIC}")
        #----------------------------
        start_time = time.time()  
        while True:  
            try:  
                with sqlite3.connect(FRIGATE_DB_PATH) as frigate_db_con:  
                    cursor = frigate_db_con.cursor()  
                    cursor.execute("SELECT id, label, camera, start_time, end_time, thumbnail FROM event WHERE id = ?", (event_id,))  
                    event_data = cursor.fetchone()  
                    if event_data and len(event_data) == 6:  
                        break  
            except sqlite3.Error as e:  
                logging.error(f"Error accessing Frigate database: {e}")  
            if time.time() - start_time > 30:  
                return  
            time.sleep(1)  

        if event_data and len(event_data) == 6:  
            start_time = time.time()  
            while True:  
                try:  
                    with sqlite3.connect(EVENTS_DB_PATH) as events_db_con:  
                        setup_database(events_db_con)  
                        cursor = events_db_con.cursor()  
                        cursor.execute(  
                            """  
                            INSERT INTO event (  
                                id, label, camera, start_time, end_time, thumbnail,   
                                sub_label, snapshot_path, video_path  
                            )   
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)  
                            ON CONFLICT (id) DO UPDATE SET  
                                sub_label = excluded.sub_label  
                            """,  
                            (event_data[0], event_data[1], event_data[2], event_data[3], event_data[4], event_data[5], recognized_name, out_image_path, video_path)  
                        )
                        events_db_con.commit()  
                        break  
                except sqlite3.Error as e:  
                    logging.error(f"Error accessing Events database: {e}")  
                if time.time() - start_time > 30:  
                    return  
                time.sleep(1)  
        current_time = time.time()
        period = current_time - start_time
        logging.info(f"Updating dataset took {period} seconds.")
         

    else:  
        logging.error("File was not created in time.")  

# Enhanced function to wait for file creation and ensure it's ready to be read  
def wait_for_file_creation(file_path, timeout=10, check_interval=0.5):  
    start_time = time.time()  
    while time.time() - start_time < timeout:  
        if os.path.exists(file_path):  
            try:  
                with open(file_path, 'rb') as f:  
                    f.read()  
                return True  
            except IOError:  
                pass  
        time.sleep(check_interval)  
    logging.error(f"Timeout reached. File not found or not ready: {file_path}")  
    return False  

# Function to set up the database tables if they do not exist  
def setup_database(connection):  
    cursor = connection.cursor()  
    cursor.execute(EVENT_TABLE_SCHEMA)  
    connection.commit()  

def fetch_best_snapshot(event_id, base_url= FRIGATE_SERVER_ADDRESS):  
    # Construct the URL for accessing the best snapshot  
    snapshot_url = f"{base_url}/api/events/{event_id}/snapshot.jpg"  
    
    # Make the HTTP request to fetch the image  
    response = requests.get(snapshot_url)  
    
    if response.status_code == 200:  
        # Load the response content as an image  
        image = Image.open(BytesIO(response.content))  
        return image  
    else:  
        print(f"Failed to fetch snapshot. Status code: {response.status_code}")  
        return None  

def save_snapshot_image(image, file_path):  
    try:  
        # Ensure the directory exists  
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        # Save the image  
        image.save(file_path)  
        print(f"Snapshot saved at: {file_path}")  
    except Exception as err:  
        print(f"Failed to save image: {err}") 

if __name__ == "__main__":  
    client = mqtt_client.Client()  
    client.on_connect = on_connect  
    client.on_message = on_message  

    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  
    client.connect(BROKER_HOST, BROKER_PORT, 60)  

    logging.info("Starting MQTT client loop")  
    client.loop_forever()