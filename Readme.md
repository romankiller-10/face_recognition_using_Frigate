# Real-time Facial Recognition using Facenet  

This project uses Frigate, MQTT, FaceNet, and SQLite for real-time facial recognition. Below is the developer's guide to set up and run the project.  

## Developer's Guide  

### Setting up the Development Environment  
1. **Clone the repository**  
2. **Make sure Python (>3.8.10) is installed.**  

### Running the Project  

#### Development with Docker & Local  

1. **Deploy the project in a virtual environment**:  
   ```shell  
   python local.py
2. **Run the server in a Docker container**:  
   
    This will run the FaceNet server in a Docker container. Start the server with:  
    ```shell  
    sudo docker compose up --build

#### File Organization 🗄️

```shell
├── Real-time-face-recognition-Using-Facenet (Current Directory)  
    ├── encodings  
    ├── architecture.py  
    ├── detect.py  
    ├── train_data_augmented.py  
    ├── local.py  
    ├── facenet_keras_weights.h5  
    ├── requirements.txt  
    ├── Faces  
        ├── Azam  
        ├── Winnie  
        └── JackieChan  
    └── Readme.md  
    └── Dockerfile  
    └── docker-compose.yml
```

#### Obtaining the encoding dict for face images using facenet.

1. Place the new image of persons in the Faces folder.
2. Run the following script.

   ```
	python train.py
This command will create the data_augmented_encodings.pkl file in encodings folder.

#### How to swap the facenet.
1. The current model architecture of facenet is InceptionResNetV2 as in architecture.py. 
If you want to try new faceent, then change this with new one.
2. Change also the weight file.
The weight of the current model is in facenet_keras_weights.h5.
In constants.py, you can set the model_weight_path.
