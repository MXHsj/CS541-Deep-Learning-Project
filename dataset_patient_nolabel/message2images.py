# =======================================================================
# file name:    rostopic2images.py
# description:  covert US images in rostopic to individual image files,
#               this script only needs to be run once
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import os
import cv2
import time
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Message2Images():
  def __init__(self) -> None:
    rospy.init_node('convert_to_frame', anonymous=True)
    rospy.Subscriber('/Clarius/US', Image, callback=self.clarius_us_cb)
    self.IMG_ORIG_WIDTH = 640       # image width for display
    self.IMG_ORIG_HEIGHT = 480      # image height for display
    self.frame_count = 0
    self.rate = rospy.Rate(10)
    self.loop()

  def crop_frame(self, image: np.array) -> np.array:
    x = 100                     # lateral start
    y = 0                       # axial start
    w = self.IMG_ORIG_WIDTH-x   # width
    h = self.IMG_ORIG_HEIGHT    # height
    return image[y:y+h, x:x+w]

  def save_frame(self, frame2save: np.array) -> None:
    save_path = os.path.dirname(__file__) + 'image/'
    file_tag = 'xm-frame' + str(self.frame_count)
    file_name = save_path + file_tag+'.jpg'
    cv2.imwrite(file_name, frame2save)
    return file_name

  def clarius_us_cb(self, msg: Image) -> None:
    bmode_raw = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    file_name = self.save_frame(bmode_raw)
    self.frame_count += 1
    print(f'save to: {file_name}, frame count: {self.frame_count}')
    time.sleep(0.06)  # US publish rate: 20hz, sleep here to downsample

  def loop(self):
    while not rospy.is_shutdown():
      self.rate.sleep()


if __name__ == '__main__':
  converter = Message2Images()
