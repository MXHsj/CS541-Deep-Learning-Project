# lung-ultrasound image segmentation
Residual U-Net multi-class segmentation

## Usage
- run ```train.py``` to train the model. To train the encoder only, set the ```isTrainEncoder``` flag to ```True```. Set ```isTrainEncoder``` to ```False``` to train the whole network for segmentation
- run ```infer.py``` to repeatedly predict on one input image (pseudo real-time) 