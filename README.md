#Installation

conda create --name python_3.9-env python=3.9
activate python_3.9-env
pip install opencv-python
pip install cmake
conda install -c conda-forge dlib
pip install face_recognition

# Need for face emotion training
conda install tensorflow
pip install tensorflow

conda install keras
pip install keras


# Face Landmarks
pip install pillow
