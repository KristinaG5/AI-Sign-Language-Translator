
# Sign Language Recognition and Translation

**Contents**:

1. [Overview](#overview)
2. [Installation](#installation)
3. [How to run](#steps)

# Overview
The Queen's University Belfast (2020-2021) - Final Year Project

**Supervised by:** Dr. J. Bustard

**Project**: A research based dissertation project with emphasis on finding a comprehensive solution to translate British Sign Language. The project will focus on dataset collection, video processing, visualisation and machine learning with a focus on reliability and accuracy in the final solution.

## Prerequisites
- [Python 3.8](https://www.python.org/)
- Nvidia GPU

## Installation for Ubuntu
1. Install Nvidia driver version 450
    1. Go to Software & updates
    2. Select nvidia-driver-450 from drivers tab
2. Install Cuda
```
$ sudo apt-get install nvidia-cuda-toolkit
```
3. Install [cuDNN 7.5.1](https://developer.nvidia.com/cudnn): Download runtime, developer and code library for version 7.5.1 for CUDA 10.1. From the directory of these files run:
```
$ sudo dpkg -i libcudnn7_7.5.1.10-1+cuda10.1_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.5.1.10-1+cuda10.1_amd64.deb
$ sudo dpkg -i libcudnn7-doc_7.5.1.10-1+cuda10.1_amd64.deb
```
4. Verify CUDA & cudNN install
```
$ nvcc  --version
$ cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
5. Install dependencies
```
$ pip install -r requirements.txt
```
6. Update submodules
```
$ git submodule update --init --recursive
``` 
7. Download Frankmocap data (`extra-data/`)from https://drive.google.com/file/d/133FiIgbH_q3p5G2mm55OumTH89Zq4qzm/view?usp=sharing 

8. Unzip `extra-data/` to preprocessing/frankmocap and add them to `preprocessing/frankmocap` directory.
9. Install Chromedriver (optional)


## Installation for Windows
1. Install latest Nvidia graphics driver
2. Install dependencies
```
$ pip install -r requirements.txt
```
3. Update submodules
```
$ git submodule update --init --recursive
``` 
4. Download Frankmocap data from https://drive.google.com/file/d/133FiIgbH_q3p5G2mm55OumTH89Zq4qzm/view?usp=sharing

5. Unzip extra_data to preprocessing/frankmocap
6. Install Chromedriver (optional)

## Install Openpose (optional)
1. Clone OpenPose and cd into directory
```
$ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
$ cd openpose
```
2. Install dependencies
```
$ sudo bash ./scripts/ubuntu/install_deps.sh
```
3. Install OpenCV
```
$ sudo apt-get install libopencv-dev
```
4. Install CMake
```
$ sudo apt-get install cmake-qt-gui
```
5. Add following to the end of .bashrc
```
export PYTHON_EXECUTABLE=/usr/bin/python3.8
export PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
export PYTHONPATH=/usr/local/python/:$PYTHONPATH
``` 
6. Set gcc version to 8
```
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
```
7. Install OpenPose by following the [installation guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/README.md#cmake-configuration)

# Steps

1. Gather data
2. Preprocess 
3. Train
4. App

## Gather data
All scripts for fetching data can be found in the `dataset` directory.

To build the validation dataset:
1. Run `webscraper/sentence_url_fetcher.py` to fetch the sentence urls from http://digital-collections.ucl.ac.uk/R/?local_base=BSLCP
2. Run `webscraper/sentence_clip_fetcher.py` to download the sentence videos

To produce validation sentence videos:
1. Run `sentence_file_reader.py` to split the validation videos into individual sentences based on the EAF files

To build the training dataset:
1. Run `webscraper/signbank_url_fetcher.py` to fetch A-Z word URLs from https://bslsignbank.ucl.ac.uk/dictionary/search/?
2. Run `webscraper/signbank_video_url_fetcher.py` to fetch the video links for each letter
3. Run `signbank_clip_fetcher.py` to download the videos
4. Run `webscraper/signbsl_url_fetcher.py` to fetch A-Z word URLs from https://www.signbsl.com/dictionary/
5. Run `webscraper/signbsl_video_url_fetcher.py` to fetch the video links for each letter
6. Run `signbsl_clip_fetcher.py` to download the videos
7. Run `dataset_trim.py` to clean and reduce the dataset


## Preprocessing
```preprocessing/run_preprocessing.py```: This runs and creates a JSON file containing the coordinates for a folder of videos

Arguments:
- `--folder_path`: Path to folder of folders where each subfolder represents a class:
    ├── parent_folder
        ├── cat
            ├── cat1.mp4
            ├── cat2.mp4
            └── cat3.mp4
        └── dog
            ├── dog1.mp4
            ├── dog2.mp4
            └── dog3.mp4
- `--output_folder`: Path to the location to save the json files produced:
    ├── parent_folder
        ├── cat.json
        └── dog.json
- `--normalisation_type`: Normalisation technique (choice between ratio and angle)
- `--method`: Select pose estimation library to extract pose (supports FrankMocap and Openpose)
- `--frankmocap_dir`: Path to the frankmocap models dir
- `--openpose_dir`: Path to the openpose build dir
- `--openpose_models_dir`: Path to the openpose models dir
- `--fps`: Sampling rate for extracting poses


## Training
```training/train.py```: This trains the preprocessed data on sklearn KNN classifier using Dynamic Time Warping metric to produce model weights used in the app

Arguments:
- `--folder_path`: Path to folder of JSONs produced in preprocessing step
- `--dtw_method`: Dynamic Time Warping calculcation method (supports standard and FastDTW)
- `--n_neighbors`: Set N neighbors for KNN
- `--max_warping_window`: Size of warping window
- `--evaluate`: Scores model after training
- `--train_size`: Sets train/test spit size
- `--model_path`: Path to output model

Pre-trained models for the small and large dataset can be downloaded from https://drive.google.com/drive/folders/1L5lPZk-Of4qfPNYXYNlIMDUX5SsEtc8S?usp=sharing

## App
```application/app.py```: User interface to predit on a video given using the model produced in training.
