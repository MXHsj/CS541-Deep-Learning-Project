# =======================================================================
# file name:    infer.py
# description:  real-time inference as ROS node
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-12-03
# version:
# =======================================================================
import cv2
import time
import numpy as np
import torch
from unet.model import UNet
import torch.nn.functional as F
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from utils.vis import array2tensor, tensor2array


class LungUltrasoundSegmentServer:
  ''' subscribe to US image topic & perform segmentation
  '''
  disp_window_name = "stacked display"

  def __init__(self, isVis=True, freq=60) -> None:
    # ========== params ==========
    self.IMG_WIDTH = 256            # image width for inference
    self.IMG_HEIGHT = 256           # image height for inference
    self.IMG_DISP_WIDTH = 640       # image width for display
    self.IMG_DISP_HEIGHT = 480      # image height for display
    self.isVis = isVis              # turn on/off rt visualization
    # ============================

    # ========== load model ===========
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {self.device}")
    self.net = UNet(n_channels=1, n_classes=3, bilinear=False)
    self.net.to(device=self.device)
    self.net.load_state_dict(torch.load('checkpoints/checkpoint_epoch5.pth'))
    # print(net.eval())
    # =================================

    # ========== pre-allocation ===========
    self.bmode_raw = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH, 1), dtype=np.uint8)
    self.bmode_rgb = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH, 3), dtype=np.uint8)  # for creating overlay
    self.msk_bright = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH), np.uint8)
    self.msk_dark = np.zeros((self.IMG_DISP_HEIGHT, self.IMG_DISP_WIDTH), np.uint8)
    # =====================================

    # ========== initialize ROS node ==========
    rospy.init_node('Lung_US_Segmentation_Server', anonymous=True)
    self.US_sub = rospy.Subscriber('Clarius/US', Image, self.clarius_us_cb)
    self.rate = rospy.Rate(freq)    # loop rate
    # =========================================

  def clarius_us_cb(self, msg: Image) -> None:
    bmode_msg = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    self.bmode_raw = np.expand_dims(bmode_msg, axis=-1)
    self.bmode_rgb = cv2.cvtColor(bmode_msg, cv2.COLOR_GRAY2RGB)

  def vis(self, left=None, right=None):
    if left is not None and right is not None:
      stack = np.hstack((left, right))
    else:  # create 4x4 layout
      bottom = np.hstack((self.bmode_rgb, self.bmode_rgb))
      top = np.hstack((cv2.cvtColor(self.msk_bright, cv2.COLOR_GRAY2RGB).astype(np.uint8),
                      cv2.cvtColor(self.msk_dark, cv2.COLOR_GRAY2RGB).astype(np.uint8)))
      stack = cv2.addWeighted(bottom, 0.4, top, 0.6, 1.0)
      cv2.imshow(self.disp_window_name, stack)

  def run_inference(self) -> None:
    ''' perform real-time inference, blocking behavior
    '''
    try:
      if self.isVis:
        cv2.namedWindow(self.disp_window_name)
      while not rospy.is_shutdown():
        start = time.perf_counter()
        input = cv2.resize(self.bmode_raw, (self.IMG_HEIGHT, self.IMG_WIDTH))
        input = array2tensor(input, device=self.device)
        pred = self.net(input)
        pred = tensor2array(pred)
        self.msk_bright = cv2.resize(pred[1, :, :]*255, (self.IMG_DISP_WIDTH, self.IMG_DISP_HEIGHT))
        self.msk_dark = cv2.resize(pred[2, :, :]*255, (self.IMG_DISP_WIDTH, self.IMG_DISP_HEIGHT))
        print(f'time elapsed: {time.perf_counter()-start} sec')  # benchmarking
        key = cv2.waitKey(2)
        if key == ord('q'):
          break
        self.vis()
    except Exception as e:
      print(f'run inference error: ', e)
    finally:
      if self.isVis:
        cv2.destroyAllWindows()


if __name__ == "__main__":
  server = LungUltrasoundSegmentServer()
  server.run_inference()
