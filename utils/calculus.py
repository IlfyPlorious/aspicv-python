from utils.quarter_class import Quarter


def compute_hour_from_quarter(quarter: Quarter):
    # strongest line from the hough space
    strongest_line = quarter.lines[0]
    angle = strongest_line[1]

    if quarter.quarter == 1:
        # between 12 and 3 we can divide
        # the quarter into 3 sections of 30 degrees
        if angle < 30:
            return 2
        elif angle < 60:
            return 1
        else:
            return 12

    # here the theta will be > 90 degrees as
    # our reference is the Ox axis on the bottom
    # we will subtract theta from 180 to get the
    # angle from the top Ox axis
    # so that 0 degrees is 3 o'clock (meaning a theta of 90)
    if quarter.quarter == 2:
        angle = 180 - angle
        if angle < 30:
            return 3
        elif angle < 60:
            return 4
        else:
            return 5

    # here we can reference the angle as in quarter 1
    # just keeping in mind that for 0 degrees we have
    # 9 o'clock
    if quarter.quarter == 3:
        if angle < 30:
            return 8
        elif angle < 60:
            return 7
        else:
            return 6

    # here, the axes are again switch so
    # we will subtract from 180
    if quarter.quarter == 4:
        # check for the case where angle is 0,
        # and there's no need to check
        if 90 <= angle < 180:
            angle = 180 - angle

        if angle < 30:
            return 9
        elif angle < 60:
            return 10
        else:
            return 11


def compute_minutes_from_quarter(quarter: Quarter):
    # strongest line from the hough space
    strongest_line = quarter.lines[0]
    angle = strongest_line[1]

    if quarter.quarter == 1:
        # between 12 and 3 we can divide
        # the quarter into 15 sections of 6 degrees
        minute = 15
        for check_angle in range(6, 90, 6):
            if angle < check_angle:
                return minute
            else:
                minute -= 1

        return minute

    # here the theta will be > 90 degrees as
    # our reference is the Ox axis on the bottom
    # we will subtract theta from 180 to get the
    # angle from the top Ox axis
    if quarter.quarter == 2:
        angle = 180 - angle
        minute = 15
        for check_angle in range(6, 90, 6):
            if angle < check_angle:
                return minute
            else:
                minute += 1

        return minute

    # here we can reference the angle as in quarter 1
    # just keeping in mind that for 0 degrees we have
    # 45 minutes
    if quarter.quarter == 3:
        minute = 45
        for check_angle in range(6, 90, 6):
            if angle < check_angle:
                return minute
            else:
                minute -= 1

        return minute

    # here, the axes are again switch so
    # we will subtract from 180
    if quarter.quarter == 4:
        # check for the case where angle is 0,
        # and there's no need to check
        if 90 <= angle < 180:
            angle = 180 - angle

        minute = 45
        for check_angle in range(6, 90, 6):
            if angle < check_angle:
                return minute
            else:
                minute += 1

        return minute
