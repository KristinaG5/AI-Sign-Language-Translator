def get_middle_frame(videos):
    frames = []
    for video in videos:
        middle_frame = video[int(len(video) / 2) - 1]
        frames.append(middle_frame)
    return frames
