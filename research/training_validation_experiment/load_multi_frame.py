def flatten_video(video):
    values = []
    for frame in video:
        for point in frame:
            values.append(point)
    return values


def flatten_videos(videos):
    out = []
    for video in videos:
        values = flatten_video(video)
        out.append(values)

    return out


def get_max_video_length(videos):
    max_video_length = 0

    for video in videos:
        if len(video) > max_video_length:
            max_video_length = len(video)

    return max_video_length


def pad_video(video, max_video_length):
    if len(video) < max_video_length:
        video.extend([0] * (max_video_length - len(video)))
    return video


def pad_videos(videos):
    max_video_length = get_max_video_length(videos)

    for i in range(len(videos)):
        videos[i] = pad_video(videos[i], max_video_length)

    return videos
