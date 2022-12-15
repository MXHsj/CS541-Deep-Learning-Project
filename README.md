# lung-ultrasound image segmentation
U-Net multi-class segmentation

## Usage
- run ```train.py``` to train the model
- run ```infer.py``` to repeatedly predict on one input image (pseudo real-time)
- run ```infer_node.py``` to predict on real-time streamed ultrasound images. This requires ROS, ROS network environment and Clarius ultrasound probe.