# Disable the warning  
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses every log message except errors.  

import tensorflow as tf  
from architecture import *  
import os  
import cv2  
import mtcnn  
import pickle  
import numpy as np  
from sklearn.preprocessing import Normalizer  
from constants import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

# TPU and GPU/CPU setup  
try:  
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection  
    print('Running on TPU ', tpu.master())  
except ValueError as e:  
    tpu = None  
    print('Not connected to a TPU runtime; configuring TensorFlow to use a GPU or CPU.')  

if tpu:  
    tf.config.experimental_connect_to_cluster(tpu)  
    tf.tpu.experimental.initialize_tpu_system(tpu)  
    strategy = tf.distribute.TPUStrategy(tpu)  
else:  
    strategy = tf.distribute.MirroredStrategy()  # Use MirroredStrategy to run on multiple GPUs or a single GPU  
    print("Using MirroredStrategy with available GPUs or CPU.")  

# Paths and Variables  
face_data = 'Faces/'  # Directory where face images are stored  
required_shape = REQUIRED_IMAGE_SIZE  # Required shape for face images  

with strategy.scope():  # Create the model under the strategy scope  
    face_encoder = InceptionResNetV2()  # Model for encoding faces  
    path = MODEL_WEIGHTS_PATH  # Path to the pretrained weights  
    face_encoder.load_weights(path)  

face_detector = mtcnn.MTCNN()  
encodes = []  # List to hold face encodings  
encoding_dict = {}  # Dictionary to map person's name to their face encoding  
l2_normalizer = Normalizer('l2')  # Normalizer for encoding  
augmenter = ImageDataGenerator(  
    rotation_range=15,  # Rotate images by a maximum of 15 degrees  
    width_shift_range=0.1,  # Shift images width-wise within the range 10%  
    height_shift_range=0.1,  # Shift images height-wise within the range 10%  
    shear_range=0.1,  # Shear images within a range of 10%  
    zoom_range=0.1,  # Zoom-in or Zoom-out on images within a range of 10%  
    horizontal_flip=True,  # Randomly flip images horizontally  
    fill_mode='nearest'  # Fill pixels using nearest filling mode  
)  

def normalize(img):  
    """Normalize the image by subtracting the mean and dividing by the standard deviation."""  
    mean, std = img.mean(), img.std()  
    return (img - mean) / std  

def save_augmented_images(augment_dir, img, filename_prefix):  
    """  
    Save augmented versions of the given image to the specified directory.  

    Parameters:  
    - augment_dir (str): Directory where augmented images will be saved.  
    - img (numpy array): Image to augment.  
    - filename_prefix (str): Prefix for the filenames of augmented images.  
    """  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB  
    img_expanded = np.expand_dims(img_rgb, 0)  # Expand dimensions to fit the augmenter requirements  
    aug_iter = augmenter.flow(img_expanded)  # Create an iterator for augmenting the image  
    for i in range(5):  # Generate 5 augmentations per image  
        aug_img = next(aug_iter)[0].astype(np.uint8)  # Get next augmented image and convert to uint8  
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving  
        augmented_image_path = os.path.join(augment_dir, f"{filename_prefix}_aug_{i}.jpg")  
        cv2.imwrite(augmented_image_path, aug_img_bgr)  # Save the augmented image  
        print(f"New augmented image saved at: {augmented_image_path}")  # Log the augmented image path  

if __name__ == "__main__":  

    for face_names in os.listdir(face_data):  
        person_dir = os.path.join(face_data, face_names)  # Directory of current person  
        augment_dir = os.path.join(person_dir, "augmented")  # Subdirectory for augmented images  

        if not os.path.isdir(person_dir):  
            continue  

        os.makedirs(augment_dir, exist_ok=True)  # Create augmented directory if it doesn't exist  

        augmented_images_exist = len(os.listdir(augment_dir)) > 0  # Check if augmented images already exist  

        if not augmented_images_exist:  # Only create augmented images if none exist  
            for image_name in os.listdir(person_dir):  
                image_path = os.path.join(person_dir, image_name)  
                img_BGR = cv2.imread(image_path)  

                if img_BGR is None or os.path.isdir(image_path):  
                    continue  

                # Save augmented images  
                save_augmented_images(augment_dir, img_BGR, os.path.splitext(image_name)[0])  

        print(f"Processing person: {face_names}")  

        for image_name in os.listdir(person_dir) + os.listdir(augment_dir):  
            image_path = os.path.join(person_dir, image_name) if image_name in os.listdir(person_dir) else os.path.join(augment_dir, image_name)  
            img_BGR = cv2.imread(image_path)  

            if img_BGR is None:  
                continue  

            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)  # Convert image to RGB  
            x = face_detector.detect_faces(img_RGB)  # Detect faces in the image  

            if not x:  
                continue  

            x1, y1, width, height = x[0]['box']  
            x1, y1 = abs(x1), abs(y1)  
            x2, y2 = x1 + width, y1 + height  
            face = img_RGB[y1:y2, x1:x2]  # Extract the face from the image  

            face = normalize(face)  # Normalize the face  
            face = cv2.resize(face, required_shape)  # Resize the face to the required shape  
            face_d = np.expand_dims(face, axis=0)  # Expand dimensions to fit the model requirements  
            with strategy.scope():  # Predict face encoding within the strategy scope  
                encode = face_encoder.predict(face_d)[0]  # Get the encoding for the face  
            encodes.append(encode)  # Append the encoding to the list  

        if encodes:  
            encode = np.sum(encodes, axis=0)  # Sum of encodings  
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]  # Normalize the sum of encodings  
            encoding_dict[face_names] = encode  # Map person's name to their face encoding  
            encodes = []  # Reset encodes for next person  

    print("Getting encoding dict....")  # Log before getting the encoding dictionary  

    # Save encodings to a file  
    encodings_path = ENCODINGS_PATH  
    os.makedirs(os.path.dirname(encodings_path), exist_ok=True)  
    with open(encodings_path, 'wb') as file:  
        pickle.dump(encoding_dict, file)  # Save the dictionary containing the encodings  
        print(f"Encodings saved at: {encodings_path}")  # Log the path where encodings are saved