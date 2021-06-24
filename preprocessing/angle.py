import math


def get_pair(data, index):
    """Gets x and y pair from JSON

    Arguments
    ---------
    data : array
        Individual frame points

    index : int
        Point to describe the joint end (visualise.csv)

    Returns
    -------
    Array of two points
    """
    return [data[index * 2], data[(index * 2) + 1]]


def calculate_angle(pointA, pointB):
    """Calculates angle

    Arguments
    ---------
    pointA : array
        Array of origin point

    pointB : array
        Array of destination point

    Returns
    -------
    Calculated angle
    """
    x_change = pointB[0] - pointA[0]
    y_change = pointB[1] - pointA[1]

    if x_change == 0:
        return 0

    return math.degrees(math.atan(y_change / x_change))


def get_joints(video, frame_index, points):
    """Applies angle joint calculation

    Arguments
    ---------
    video : dictionary
        Dictionary containing video information

    frame_index : int
        Index of current frame

    points : int
        Origin and destination points describing each joint

    Returns
    -------
    Array
    """
    angles = []
    for point in points:
        body_type, origin_index, destination_index = point.split(",")
        frame = video[body_type][frame_index]
        origin = get_pair(frame, int(origin_index))
        destination = get_pair(frame, int(destination_index))
        angles.append(calculate_angle(origin, destination))
    return angles


def calculate_joint_angles(video):
    """Applies angle joint normalisation to pose

    Arguments
    ---------
    video : dictionary
        Dictionary containing video information

    Returns
    -------
    Array of frame points
    """
    with open("visualise.csv") as f:
        points = [line.strip() for line in f.readlines()[1:]]

    frames = []
    for i in range(len(video["body"])):
        frames.append(get_joints(video, i, points))

    return frames
