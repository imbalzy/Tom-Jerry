# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FFMpegWriter
# import matplotlib.image as mgimg
# import os
# import ffmpeg
# from matplotlib import animation
#
# folder = '02_Dec_2021_03_14_05'
# path = "../results/"+ folder + "/figures"
# save_path = "../results/"+ folder + "/video/"
# if not os.path.exists(os.path.join(os.getcwd(), save_path)):
#     os.mkdir(os.path.join(os.getcwd(), save_path))
#
# listing = os.listdir(os.path.join(os.getcwd(), path))
#
# writer = FFMpegWriter(fps=15)
# fig = plt.figure()
# im = plt.imshow(np.zeros((20,20)))
#
# FFwriter=animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
#
# with writer.saving(fig,'anim.mp4', 100):
#     for file in listing:
#         img = mgimg.imread(path + "/" + file)
#         im.set_data(img)
#         writer.grab_frame()

import cv2
import os

folder = '02_Dec_2021_07_01_02'
image_folder = "../results/"+ folder + "/figures"
video_folder = "../results/"+ folder + "/video/"
if not os.path.exists(os.path.join(os.getcwd(), video_folder)):
    os.mkdir(os.path.join(os.getcwd(), video_folder))


video_name = video_folder + '/example0.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img[8] == '0']
images.sort()
images = images[1:]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 2, (width, height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

video_name = video_folder + '/example1.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img[8] == '1']
images.sort()
images = images[1:]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 2, (width, height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()


