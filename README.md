# Data Flow System for Video and Image Analysis

Video and image analysis using deep learning has become ubiquitous across many different applications. However, traditional architectures are optimized for accuracy regardless of their inference time. This characteristic can potentially impair real world applications that have either computational resources limitations or time sensitivity. We propose a data flow system composed of different operators that allows for the inference time to be automatically adjusted according to the data being analyzed. It gracefully adjusts its accuracy (images) and frame sampling rate (video) in order to maintain a smooth output flow.

## Installation

* Python 3.6
* [PyTorch 1.1 and Torchvision](https://pytorch.org)


### Python libraries:

* [OpenCV](https://pypi.org/project/opencv-python/)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* multiprocessing

### Image classification: 

We used the model described in the paper [Band-limited Training and Inference for Convolutional Neural Networks](https://icml.cc/Conferences/2019/Schedule?showEvent=4555) to automatically adjust the compression rate for a faster inference time, while gracefully degrading the accuracy for image classification. Please, follow the instructions on their [repository](https://github.com/adam-dziedzic/bandlimited-cnns) replacing their `main.py` file for the `inference.py` file found in the classification folder of this repository.

Checkpoints for four different models can be downloaded running the `download_checkpoints.sh` bash script in the classification folder. The checkpoint files should be placed in the same folder as the `inference.py` file.

### Object detection:

We used the model proposed in the paper [Yolov3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) for object detection, which is among the state of the art architectures for this purpose. Our code is based on [Erik Lindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3)'s implementation. Run the `download_weights.sh` bash script in the detection/config folder to download the Yolo weights file used in the model.

Sample videos used in the experiments can be downloaded running the `download_videos.sh` bash script in the detection folder.

## Usage

### Image classification:

The image classification requires three arguments: the dataset, the checkpoint with the trained model and the output display boolean. The following is an example using one of the checkpoints provided:

`inference.py --dataset='cifar10 --resume='' --output_display='False'`


### Object detection:

The object detection only requires three arguments: the input video file path, the output video file path and the output display boolean. The following is an example using one of the videos downloaded in the installation step:

`detection.py --input='input_video/Samsung_UHD_Soccer_Barcelona_Atletico_Madrid.ts' --output='output_video/soccer_output.avi'  --display_output='False'`
