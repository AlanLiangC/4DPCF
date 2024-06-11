import numpy as np

LiDAR_Hardware_parameter = {
    'kitti-od' : dict(
        vertical_fov_min=-24.9,
        vertical_fov_max = 2.0,
        num_points_per_beam=200,
        num_beams=64
    ),
    'ArgoVerse' : dict(
        vertical_fov_min=-30,
        vertical_fov_max = 10,
        num_points_per_beam=500,
        num_beams=32
    )
}

def generate_custom_style_point_cloud(vertical_fov_min,
                                      vertical_fov_max,
                                      num_points_per_beam, 
                                      num_beams):

    vertical_fov = np.linspace(np.radians(vertical_fov_min), np.radians(vertical_fov_max), num_beams)

    horizontal_fov_min = 0
    horizontal_fov_max = 360
    horizontal_fov = np.linspace(np.radians(horizontal_fov_min), np.radians(horizontal_fov_max), num_points_per_beam)

    distances = np.random.uniform(0, 50, (num_beams, num_points_per_beam)) 
    azimuths = np.tile(horizontal_fov, (num_beams, 1))
    elevations = np.tile(vertical_fov.reshape(-1, 1), (1, num_points_per_beam))

    x = distances * np.cos(elevations) * np.cos(azimuths)
    y = distances * np.cos(elevations) * np.sin(azimuths)
    z = distances * np.sin(elevations)

    point_cloud = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    return point_cloud

if __name__ == '__main__':

    point_cloud = generate_custom_style_point_cloud(**LiDAR_Hardware_parameter['ArgoVerse'])
    print(point_cloud.shape)
