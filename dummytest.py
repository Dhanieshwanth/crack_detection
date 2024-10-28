import cv2

d  = cv2.imread("crack_segmentation_dataset/test/images/image1.jpg")
d1 = cv2.imread("crack_segmentation_dataset/test/images/CFD_001.jpg")

print(d.shape)
print(d1.shape)