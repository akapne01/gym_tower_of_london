class BallCoordinates:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class AllBallCoordinates:
    def __init__(self, x, tol_height, tol_aspect_ratio):
        length = tol_height / tol_aspect_ratio
        self.x = x
        self.height = tol_height
        self.length = length
        self.radius = (length - tol_height) / 5
        increment = (length - tol_height) / 2
        self.y = x + tol_height - increment - self.radius
        # contains x, y, and z coordinates of each rod and ball possible
        # 00 will put the ball on rod 0, position 0
        # 32 will put the ball on rod 3 position 2
        self.coordinate_matrix = self._create_coordinates()

    def _create_coordinates(self):
        add_x = self.length - self.height
        r0_b0 = BallCoordinates(self.x, self.y, self.radius)
        r1_b0 = BallCoordinates(self.x + add_x, self.y, self.radius)
        r1_b1 = BallCoordinates(self.x + add_x, self.y - self.radius * 2, self.radius)
        r2_b0 = BallCoordinates(self.x + add_x * 2, self.y, self.radius)
        r2_b1 = BallCoordinates(self.x + add_x * 2, self.y - self.radius * 2, self.radius)
        r2_b2 = BallCoordinates(self.x + add_x * 2, self.y - self.radius * 4, self.radius)
        matrix_dict = {'00': r0_b0, '10': r1_b0, '11': r1_b1, '20': r2_b0, '21': r2_b1, '22': r2_b2}
        return matrix_dict


if __name__ == '__main__':
    co = AllBallCoordinates(x=112.5, tol_height=250.0, tol_aspect_ratio=2 / 3)
    print(co.coordinate_matrix)
    a = co.coordinate_matrix.get("22")
    print(a.x)
    print(a.y)
    print(a.r)
