import math

from config import (
    ALTITUTE_BS,
    ALTITUTE_VEHICLE,
    BEAMS_AZIMUTH,
    BEAMS_ELEVATION,
    FREQUENCY,
    HPBW_BS,
    PATH_LOSS_EXPONENT,
    SLL_BS,
    SPEED_OF_LIGHT,
)

# , CELL_SCENARIO


class BaseStation:
    # cars_in_cells: "list[list[Car]]"
    def __init__(self, bs_id, coord):
        self.bs_id = bs_id
        self.x = coord[0]
        self.y = coord[1]
        self.cars_in_cells = [[], [], []]  # three lists - one for each cell
        # self.broadcast_data_size = [0, 0, 0]

    def get_path_loss(self, x_vehicle, y_vehicle):
        d_2d = math.sqrt((math.pow(x_vehicle - self.x, 2)) + (math.pow(y_vehicle - self.y, 2)))
        if d_2d == 0:
            d_2d = 0.1
        if d_2d >= 5000:
            raise ValueError("Error in distance to the base station")
        elif d_2d < 5000:
            d_3d = math.sqrt(
                (math.pow(x_vehicle - self.x, 2))
                + (math.pow(y_vehicle - self.y, 2))
                + math.pow(ALTITUTE_VEHICLE - ALTITUTE_BS, 2)
            )  # [m]
            param_k = math.pow(SPEED_OF_LIGHT / (4 * math.pi * FREQUENCY), 2)
            loss_power = param_k * math.pow(d_3d, -PATH_LOSS_EXPONENT)
            losses = -10.0 * math.log10(loss_power)  # in dB
        else:
            raise ValueError("Problem with distance between UE and GBS")
        return losses

    @staticmethod
    def get_antenna_gain(angle_azimuth_beam, angle_elevation_beam):
        # calculate the antenna gain
        gain_horizontal = -min(12 * math.pow(angle_azimuth_beam / HPBW_BS, 2), SLL_BS) + MAX_ANTENNA_GAIN_BS
        gain_vertical = max(-12 * math.pow((angle_elevation_beam) / HPBW_BS, 2), -SLL_BS)
        gain_total = gain_horizontal + gain_vertical
        return gain_total

    @staticmethod
    def get_best_beam_lobe(indx, angle):
        if indx == 0:
            lobes = BEAMS_AZIMUTH
        elif indx == 1:
            lobes = BEAMS_ELEVATION
        else:
            raise ValueError("Wrong flag for beam selection in get_Best_Beam_Lobe")
        angles = []
        for lobe in lobes:
            angles.append(abs(angle - lobe))
        angle_diff = min(angles)
        beam_id = angles.index(angle_diff)
        return angle_diff, beam_id

    @staticmethod
    def get_angle_off_boresight(beam_id, angle, indx):
        if indx == 0:
            lobes = BEAMS_AZIMUTH
        elif indx == 1:
            lobes = BEAMS_ELEVATION
        else:
            raise ValueError("Wrong flag for beam selection in get_Angle_Off_Boresight")
        angle = abs(angle - lobes[beam_id])
        return angle

    @staticmethod
    def get_azimuth(x, y, cell_id):
        quad_1 = [-30, -150, 90]
        quad_2 = [-30, -150, -270]
        quad_3 = [330, 210, 90]
        quad_4 = [-30, 210, 90]
        angle = math.degrees(math.atan2(y, x))
        if angle >= -30 and angle <= 90:
            az_angle = angle + quad_1[cell_id]
        elif angle > 90 and angle <= 180:
            az_angle = angle + quad_2[cell_id]
        elif angle > -180 and angle <= -150:
            az_angle = angle + quad_3[cell_id]
        else:
            az_angle = angle + quad_4[cell_id]
        return az_angle
