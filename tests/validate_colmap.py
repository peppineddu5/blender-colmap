"""Independent validation: python tests/validate_colmap.py <Blender fixture directory>.

Requires pycolmap only in the test environment, never in the Blender add-on.
"""
from pathlib import Path
import sys

import numpy as np
import pycolmap


root = Path(sys.argv[1])
models = (
    'test_plane_and_repeated_cameras', 'test_occlusion_and_thin_surfaces',
    'test_color_view_selection', 'test_clip_planes_and_empty_scene',
    'test_spacing_and_no_geometry', 'test_render_and_reuse', 'test_z_dense_benchmark',
)
for name in models:
    model = pycolmap.Reconstruction(str(root / name))
    assert model.is_valid(), name
    model.update_point_3d_errors()
    max_error = 0.0
    for point in model.points3D.values():
        max_error = max(max_error, point.error)
        for element in point.track.elements:
            image = model.images[element.image_id]
            observation = image.points2D[element.point2D_idx]
            projected = image.project_point(point.xyz)
            assert projected is not None, name
            error = float(np.linalg.norm(projected - observation.xy))
            assert error < .002, (name, error)
    print(f'{name}: {model.num_images()} images, {model.num_points3D()} points, '
          f'max reprojection error {max_error:.8f}px')
