#Loading packages

from models import *
from utils import *

import cv2

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from multiprocessing import Process, JoinableQueue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

#Configuration

config_path='/content/drive/My Drive/config/yolov3.cfg'
weights_path='/content/drive/My Drive/config/weights'
class_path='/content/drive/My Drive/config/coco.names'
#config_path='config/yolov3.cfg'
#weights_path='config/yolov3.weights'
#class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def detect(image):
  prev_time = time.time()
  img = Image.fromarray(image, 'RGB')
  detections = detect_image(img)
  inference_time = datetime.timedelta(seconds=time.time() - prev_time)
  print ('Inference Time: %s' % (inference_time))

  # Get bounding-box colors
  cmap = plt.get_cmap('tab20b')
  colors = [cmap(i) for i in np.linspace(0, 1, 20)]

  img = np.array(img)
  plt.figure()
  fig, ax = plt.subplots(1, figsize=(12,9))
  ax.imshow(img)

  pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
  pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
  unpad_h = img_size - pad_y
  unpad_w = img_size - pad_x

  if detections is not None:
      unique_labels = detections[:, -1].cpu().unique()
      n_cls_preds = len(unique_labels)
      bbox_colors = random.sample(colors, n_cls_preds)
      # browse detections and draw bounding boxes
      for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
          box_h = ((y2 - y1) / unpad_h) * img.shape[0]
          box_w = ((x2 - x1) / unpad_w) * img.shape[1]
          y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
          x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
          color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
          bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
          ax.add_patch(bbox)
          plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                  bbox={'color': color, 'pad': 0})
  plt.axis('off')
  # save image
  #plt.savefig(img_path.replace(".jpg", "-det.jpg"), bbox_inches='tight', pad_inches=0.0)
  plt.show()
  if detections is not None:
    return len(detections)
  else:
    return 0
  
class frame_extractor(object):
  """Load a video and extract frames placing them in a queue
  
  Args:
    video: path to the video file
    queue: queue to where the extracted images will be added.
  
  """
  def __init__(self, video, input_queue, output_queue):
    self.input_queue = input_queue
    self.output_queue = output_queue
    self.video = cv2.VideoCapture(video)
    self.sucess = True
      
  def __iter__(self):
    return self
  
  def __next__(self):
    if self.sucess:
      step = self.input_queue.get()
      if step == 1:
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.output_queue.put(image)
        print("skip 2 frames")
        return
        
      if step == 2:
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.output_queue.put(image)
        print("skip 1 frames")
        return
      
      if step >=3:
        self.success,image = self.video.read()
        self.output_queue.put(image)
        print("don't skip")
        return
      
      else:
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.success,image = self.video.read()
        self.output_queue.put(image)
        print("skip 5 frames")
        return
    
    else:
      raise StopIteration
      
class detector(object):
  """Load a frame and detect objects within it
  
  Args:
    queue: queue with the input frames.
  """
  def __init__(self, queue_input, queue_output, queue_frames):
    self.queue_input = queue_input
    self.queue_output = queue_output
    self.queue_frames = queue_frames
    
  def __iter__(self):
    return self
  
  def __next__(self):  
    if self.queue_input.qsize() != 0:
      image = self.queue_input.get()
      number = detect(image)
      self.queue_output.put(number)
      self.queue_frames.put(image)
      return
    
    else:
      raise StopIteration
      
class video_constructor(object):
  """Load a frame and detect objects within it
  
  Args:
    queue: queue with the input frames.
  """
  def __init__(self, queue_frames, video):
    self.queue_frames = queue_frames
    self.video = video
    
  def __iter__(self):
    return self
  
  def __next__(self):  
    if self.queue_frames.qsize() != 0:
      image = self.queue_frames.get()
      #image = Image.fromarray(image, 'RGB')
      video.write(image)
      return
    
    else:
      raise StopIteration