import cv2

vidcap = cv2.VideoCapture("data/validation_large_dataset/validation/validation_videos/G27n_2172.mov")
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("data/validation_large_dataset/G27n_2172_frames/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1
