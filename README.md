# lung-ultrasound image segmentation
This project aims to segment pleural line and rib shadows from lung ultrasound images.
The main branch hosts only the dataset, please refer to the following branches for detailed implementation

## branches
- UNet_binaryclass:   U-Net binary-class segmentation (baseline approach)
- UNet:               U-Net multi-class segmentation
- autoencoder-unet:   U-Net multi-class segmentation. The encoder part is separately trained on unlabeled dataset
- attention-UNet:     Attention U-Net multi-class segmentation
- resunet-UNet:       Residual U-Net multi-class segmentation
- TransUNet:          trans-UNet implementation for multi-class segmentation
