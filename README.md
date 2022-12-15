# lung-ultrasound image segmentation
U-Net multi-class segmentation. The encoder part is pretrained using unlabeled data, then the whole network is trained for segmentation

## Usage
- run ```train.py``` to train the model. To train the encoder only, set the ```isTrainEncoder``` flag to ```True```. Set ```isTrainEncoder``` to ```False``` to train the whole network for segmentation
- run ```infer.py``` to repeatedly predict on one input image (pseudo real-time)
- run ```infer_node.py``` to predict on real-time streamed ultrasound images. This requires ROS, ROS network environment and Clarius ultrasound probe.