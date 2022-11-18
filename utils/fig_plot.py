#import libraries
import cv2
from matplotlib import pyplot as plt

def show_fig(epoch, global_step, image, mask, pred_mask):
    # create figure
    fig = plt.figure(figsize=(10, 7))
    
    # setting values to rows and column variables
    rows = 1
    columns = 3

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(mask)
    plt.axis('off')
    plt.title("True mask")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    
    # showing image
    plt.imshow(pred_mask)
    plt.axis('off')
    plt.title("Pred_mask")

    plt.savefig('./result_figs/epoch_'+ str(epoch) + '_step_' + str(global_step) +'.png')

    plt.close()