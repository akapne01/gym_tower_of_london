from collections import namedtuple

from envs.custom_tol_env_dir.tol_2d.state import TolState, BallPositions

"""
Maps State to the ball positions on the pegs.
Possible ball positions are as follows:
-   - 33
-  22 32
11 21 31
First number denotes position of the red ball, 
second number denotes position if the green ball 
and third number is position of the blue ball
"""
state_ball_mapper = {
    TolState(1, 1): BallPositions(31, 32, 33),
    TolState(1, 2): BallPositions(31, 32, 11),
    TolState(1, 3): BallPositions(31, 32, 21),
    TolState(1, 4): BallPositions(31, 22, 21),
    TolState(1, 5): BallPositions(31, 11, 21),
    TolState(1, 6): BallPositions(22, 11, 21),

    TolState(2, 1): BallPositions(31, 33, 32),
    TolState(2, 2): BallPositions(31, 11, 32),
    TolState(2, 3): BallPositions(31, 21, 32),
    TolState(2, 4): BallPositions(31, 21, 22),
    TolState(2, 5): BallPositions(31, 21, 11),
    TolState(2, 6): BallPositions(22, 21, 11),

    TolState(3, 1): BallPositions(32, 32, 31),
    TolState(3, 2): BallPositions(32, 11, 31),
    TolState(3, 3): BallPositions(32, 21, 31),
    TolState(3, 4): BallPositions(22, 21, 31),
    TolState(3, 5): BallPositions(11, 21, 31),
    TolState(3, 6): BallPositions(11, 21, 22),

    TolState(4, 1): BallPositions(33, 32, 31),
    TolState(4, 2): BallPositions(11, 32, 31),
    TolState(4, 3): BallPositions(21, 32, 31),
    TolState(4, 4): BallPositions(21, 22, 31),
    TolState(4, 5): BallPositions(21, 11, 31),
    TolState(4, 6): BallPositions(21, 11, 22),

    TolState(5, 1): BallPositions(33, 31, 32),
    TolState(5, 2): BallPositions(11, 31, 32),
    TolState(5, 3): BallPositions(21, 31, 32),
    TolState(5, 4): BallPositions(21, 31, 22),
    TolState(5, 5): BallPositions(21, 31, 11),
    TolState(5, 6): BallPositions(21, 22, 11),

    TolState(6, 1): BallPositions(32, 31, 33),
    TolState(6, 2): BallPositions(32, 31, 11),
    TolState(6, 3): BallPositions(32, 31, 21),
    TolState(6, 4): BallPositions(22, 31, 21),
    TolState(6, 5): BallPositions(11, 31, 21),
    TolState(6, 6): BallPositions(11, 22, 21),
}


class ObservationSpaceCoordinates:
    """
    Class that holds matrix of all possible ball positions (x, y)
    of the Tower of London Task.
    """

    def __init__(self, x, y, height, aspect_ratio):
        """
        Creates a coordinate matrix with x, y, r positions
        :param x: Starting position x
        :param height: Height
        :param aspect_ratio: Aspect ratio
        """
        self.PositionCoordinates = namedtuple('PositionCoordinates', ['x', 'y'])
        length = height / aspect_ratio
        self.x = x
        self.height = height
        self.length = length
        self.radius = (length - height) / 5
        self.y = y
        self.coordinate_matrix = self._create_coordinates()

    def _create_coordinates(self):
        add_x = self.length - self.height
        r0_b0 = self.PositionCoordinates(self.x, self.y)
        r1_b0 = self.PositionCoordinates(self.x + add_x, self.y)
        r1_b1 = self.PositionCoordinates(self.x + add_x, self.y + self.radius * 2)
        r2_b0 = self.PositionCoordinates(self.x + add_x * 2, self.y)
        r2_b1 = self.PositionCoordinates(self.x + add_x * 2, self.y + self.radius * 2)
        r2_b2 = self.PositionCoordinates(self.x + add_x * 2, self.y + self.radius * 4)
        matrix_dict = {11: r0_b0, 21: r1_b0, 22: r1_b1, 31: r2_b0, 32: r2_b1, 33: r2_b2}
        return matrix_dict

    def get_position_coordinates(self, position):
        """
        Returns named tuple which represents the coordinates
        of particular position in the observation space.
        Position is an integer where first number represents
        the peg, and second number represents the position of
        the peg.
        :param position: integer can be 11, 21, 22, 31, 32, 33
        :return: Named tuple: PositionCoordinates which holds
        x, y and r positions to draw a ball there.
        """
        return self.coordinate_matrix.get(position)
