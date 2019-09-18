import cv2
import matplotlib.pyplot as plt

img_fname = 'ferrario1.jpg'  # assuming this is where you
# downloaded the above image
# load image
img = cv2.imread(img_fname)
if img is None:
    raise ValueError('{} is not a file'.format(img_fname))
# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# instantiate SURF
surf = cv2.xfeatures2d.SURF_create(7000)
# compute keypoints
keypoints = surf.detect(img_gray, None)
# plot keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_keypoints_rgb = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)
plt.imshow(img_keypoints_rgb)
plt.show()
