def calculate_ratio(point, min_value, length):
    """Calculates ratio

    Arguments
    ---------
    point : int
        Single x or y point

    min_value : int
        Smallest x or y point

    length : int
        Size of bounding box

    Returns
    -------
    Float
    """
    if point != 0:
        point -= min_value
        return point / length
    else:
        return 0.0


def get_coords(x, frame):
    """Calculates the smallest and largest x and y values

    Arguments
    ---------
    x : boolean
        True for x coordinate
        False for y coordinate

    frame : array
        Frame pose points

    Returns
    -------
    x and y coordinates
    """
    coord_values = [frame[i] for i in range(0 if x else 1, len(frame), 2) if frame[i] != 0]
    if coord_values:
        coord_sorted = sorted(coord_values)
        coord_min = coord_sorted[0]
        coord_max = coord_sorted[-1]
        return coord_min, coord_max
    else:
        return 0, 0


def get_box_size(frame):
    """Calculates the width and height bounding boxes

    Arguments
    ---------
    frame : array
        Frame pose points

    Returns
    -------
    x_min, y_min, box_width, box_height
    """
    x_min, x_max = get_coords(True, frame)
    y_min, y_max = get_coords(False, frame)

    box_width = x_max - x_min
    box_height = y_max - y_min

    return x_min, y_min, box_width, box_height


def get_coordinate_ratio(frame, x_min, y_min, box_width, box_height):
    """Applies ratio normalisation to pose

    Arguments
    ---------
    frame : array
        Frame pose points

    x_min : int
        Smallest x value in frame

    y_min : int
        Smallest y value in frame

    box_width : int
        Width size of the bounding box per frame

    box_height : int
        Height size of the bounding box per fram

    Returns
    -------
    Array of frame points
    """
    frame_points = []
    for i in range(len(frame)):
        if i % 2 == 0:
            point = calculate_ratio(frame[i], x_min, box_width)
        else:
            point = calculate_ratio(frame[i], y_min, box_height)
        frame_points.append(point)

    return frame_points


def get_ratio(video):
    """Applies ratio normalisation to pose

    Arguments
    ---------
    video : dictionary
        Extracted pose information from video

    Returns
    -------
    Frame points that have been normalised using ratio
    """
    frame_points = []

    for i in range(len(video["body"])):
        points = []
        for t in ["body", "left_hand", "right_hand"]:
            x_min, y_min, box_width, box_height = get_box_size(video[t][i])
            ratio = get_coordinate_ratio(video[t][i], x_min, y_min, box_width, box_height)
            points.extend(ratio)
        frame_points.append(points)

    return frame_points
