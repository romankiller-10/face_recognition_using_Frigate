# Use an official Python runtime as a parent image  
FROM python:3.8-slim  

# Upgrade pip inside the container  
RUN python -m pip install --upgrade pip  

# Install system dependencies that are required for h5py and OpenCV  
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*   

# Set environment variables for compilers  
ENV CC=gcc  
ENV CXX=g++  

# Set the working directory to /facial-server  
WORKDIR /facial-server  

# Copy the current directory contents into the container at /facial-server  
COPY . /facial-server  

# Install h5py  
RUN pip install h5py==3.11.0  

# Install any needed packages specified in requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt  

# Make port 5031 available to the world outside this container  
EXPOSE 5031  

# Run brokerv2.py when the container launches  
CMD ["python", "brokerv2.py"]