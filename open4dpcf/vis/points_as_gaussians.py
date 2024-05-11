from .vis_toos.gaussians_common import point2gaussians_ply, point2gaussians_splat

class PointsAsGuaussians(object):
    def __init__(self) -> None:
        pass

    # https://playcanvas.com/model-viewer
    @staticmethod
    def point2gaussian_ply(points, colors, path: str = None, cus_scale = 1):

        point2gaussians_ply(points = points,
                            colors = colors,
                            path = path,
                            cus_scale = cus_scale)
        
        print(f"Save points to gaussians .ply at {path}!")

    # https://antimatter15.com/splat/
    @staticmethod
    def point2gaussian_splat(points, colors, path: str = None, cus_scale = 1):

        point2gaussians_splat(points = points,
                            colors = colors,
                            path = path,
                            cus_scale = cus_scale)
        
        print(f"Save points to gaussians .splat at {path}!")
