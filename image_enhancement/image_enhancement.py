import cv2
import matplotlib.pyplot as plt
import numpy as np

def sharpen_filter(image):
    # filter kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def contrast_stretching(image):
    transformed = []

    # transform pixel in each channel
    for channel in cv2.split(image):
        map = channel.max()
        mip = channel.min()
        comp = ((channel - mip)/(map - mip))*255
        transformed.append(comp)

    # merge back all channel
    transformed_image = cv2.merge((transformed[0],transformed[1],transformed[2]))
    return transformed_image


if __name__ == '__main__':
    qingming = 'qingming_shange_tu.jpg'
    image = cv2.imread(qingming)
    ori = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply sharpen filter and contrast stretching
    sharpened = sharpen_filter(ori)
    contrast = contrast_stretching(sharpened)

    # combine both ori and transformed image
    combined = cv2.vconcat((ori,contrast.astype(np.uint8)))

    # plot image
    plt.title("Above - original\nBelow - transformed")
    plt.imshow(combined)
    plt.axis("off")
    plt.show()
