"""Gaussian Camera Rig

Create a batch of Blender cameras around the point currently being viewed.
The add-on is deliberately dependency-free so it can be installed from the
folder or from a zip file.
"""

import math
import re
import tempfile
import time
from pathlib import Path

from mathutils import Matrix, Vector

import bpy
from .raycast_export import (
    CameraFrame, ImagePixels, Observations, PointBuffer, build_geometry,
    collect_transparent_border_edges, commit_model, sample_camera,
)
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


bl_info = {
    "name": "Gaussian Camera Rig",
    "author": "Gaussian Camera Rig contributors",
    "version": (1, 16, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Gaussian Cameras",
    "description": "Generate cameras and export every scene camera as a COLMAP dataset",
    "category": "3D View",
}


COLLECTION_NAME = "GS_Cameras"
LAST_BATCH_KEY = "gs_camera_rig_last_batch"
GRID_CAMERA_SPAN_SCALE = 4.0
GLASS_BORDER_TRIANGLE = -1

# Keep the add-on keymap items so they can be removed cleanly on unregister.
addon_keymaps = []
listener_windows = set()
listener_shutdown = False
active_export = None


def _is_export_timer_event(event, timer):
    """Return whether an event should advance the export modal operator.

    Blender versions before 5.2 exposed ``Event.timer``. Blender 5.2 still
    emits ``TIMER`` events for modal operators, but some builds omit that RNA
    property, so accessing it directly raises AttributeError. When the
    property exists we retain the old identity check; when it does not, the
    operator's modal handler can only receive its own registered timer in the
    supported UI path and the event type is the compatibility signal.
    """
    if getattr(event, "type", None) != "TIMER":
        return False
    event_timer = getattr(event, "timer", None)
    return event_timer is None or event_timer == timer


def _is_spawn_hotkey(event):
    """Return whether an event matches the camera-generation shortcut."""
    return (
        event.type == "C"
        and event.value == "PRESS"
        and event.ctrl
        and event.shift
        and event.alt
    )


def _is_walk_fly_active(context):
    """Return whether this window is currently running Walk or Fly."""
    window = getattr(context, "window", None)
    for operator in getattr(window, "modal_operators", ()):
        operator_id = getattr(operator, "bl_idname", "").lower()
        if operator_id in {"view3d.walk", "view3d.fly", "view3d_ot_walk", "view3d_ot_fly"}:
            return True
    return False


def _is_walk_fly_alias(event):
    """Return whether the simpler fallback shortcut is being pressed."""
    return (
        event.type == "C"
        and event.value == "PRESS"
        and event.ctrl
        and event.shift
        and not event.alt
    )


def _scene_camera_objects(scene):
    """Return every camera object in the scene in a stable order."""
    return sorted(
        (obj for obj in scene.objects if obj.type == "CAMERA"),
        key=lambda obj: obj.name_full.casefold(),
    )


def _safe_image_name(camera_object, extension):
    """Return a filesystem-safe, stable image name for a camera object."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera_object.name_full).strip("._")
    return f"{name or 'camera'}.{extension.lower()}"


def _render_dimensions(scene):
    scale = max(float(scene.render.resolution_percentage), 1.0) / 100.0
    width = max(1, int(scene.render.resolution_x * scale))
    height = max(1, int(scene.render.resolution_y * scale))
    return width, height


def _camera_intrinsics(camera_data, scene, width, height):
    """Get intrinsics from Blender's frame (also used by the legacy projector)."""
    if camera_data.type != "PERSP":
        return None
    corners = [corner / -corner.z for corner in camera_data.view_frame(scene=scene)]
    left, right = min(c.x for c in corners), max(c.x for c in corners)
    bottom, top = min(c.y for c in corners), max(c.y for c in corners)
    fx, fy = width / (right - left), height / (top - bottom)
    return fx, fy, -left * fx, top * fy


def _colmap_pose(camera_object):
    """Return (qw, qx, qy, qz, tx, ty, tz) in COLMAP world-to-camera form.

    Blender cameras use +X right, +Y up and -Z forward.  COLMAP uses +X
    right, +Y down and +Z forward, so the axis conversion is a 180-degree
    rotation around X after transforming world coordinates into Blender's
    camera-local coordinates.
    """
    world_to_blender_camera = camera_object.matrix_world.to_3x3().normalized().transposed()
    blender_to_colmap = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    rotation = blender_to_colmap @ world_to_blender_camera
    center = camera_object.matrix_world.translation.copy()
    translation = -(rotation @ center)
    quaternion = rotation.to_quaternion()
    return (
        float(quaternion.w),
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(translation.x),
        float(translation.y),
        float(translation.z),
    )


def _mesh_material_rgb(mesh_object):
    """Return the first material color as an 8-bit RGB tuple.

    COLMAP stores point colors as integer RGB values.  Prefer the Principled
    BSDF base color when available, while keeping the material diffuse color
    as a compatibility fallback for un-noded materials.
    """
    material = next((item for item in mesh_object.data.materials if item), None)
    if material is None:
        return 128, 128, 128

    color = material.diffuse_color
    if material.use_nodes and material.node_tree is not None:
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            base_color = principled.inputs.get("Base Color")
            if base_color is not None:
                color = base_color.default_value

    return tuple(
        max(0, min(255, int(round(float(channel) * 255.0))))
        for channel in color[:3]
    )


def _mesh_object_is_exportable(obj):
    """Return whether a mesh object should contribute Blender-derived points."""
    if obj.type != "MESH" or obj.hide_render or obj.hide_viewport:
        return False
    try:
        return not obj.hide_get()
    except (AttributeError, RuntimeError):
        # hide_get can require a UI context in some Blender versions.  The
        # explicit object visibility flags above remain a safe fallback.
        return True


def _collect_mesh_points(scene, tolerance=1e-6):
    """Collect unique world-space mesh vertices with material colors.

    Points are deliberately sourced from the original mesh vertices rather
    than reconstructed from images.  The quantized position key removes
    duplicate vertices shared by objects or overlapping mesh data while
    preserving the first deterministic object's color.
    """
    points = []
    seen = set()
    scale = 1.0 / max(float(tolerance), 1e-12)

    mesh_objects = sorted(
        (obj for obj in scene.objects if _mesh_object_is_exportable(obj)),
        key=lambda obj: obj.name_full.casefold(),
    )
    for mesh_object in mesh_objects:
        world_matrix = mesh_object.matrix_world
        color = _mesh_material_rgb(mesh_object)
        for vertex in mesh_object.data.vertices:
            position = world_matrix @ vertex.co
            key = tuple(int(round(float(component) * scale)) for component in position)
            if key in seen:
                continue
            seen.add(key)
            points.append(
                {
                    "position": (float(position.x), float(position.y), float(position.z)),
                    "color": color,
                }
            )
    return points


def _project_mesh_point(camera_object, position, camera_data, scene, width, height):
    """Project a world-space point into a Blender perspective camera.

    Blender camera coordinates are +X right, +Y up, -Z forward.  COLMAP
    image coordinates are +X right, +Y down, +Z forward, so the same axis
    conversion used by ``_colmap_pose`` is applied before projection.
    """
    world_position = Vector(position)
    camera_center = camera_object.matrix_world.translation
    world_to_blender_camera = camera_object.matrix_world.to_3x3().normalized().transposed()
    local_position = world_to_blender_camera @ (world_position - camera_center)
    colmap_position = Vector((local_position.x, -local_position.y, -local_position.z))
    depth = float(colmap_position.z)
    if depth <= 1e-8:
        return None

    fx, fy, cx, cy = _camera_intrinsics(camera_data, scene, width, height)
    pixel_x = fx * float(colmap_position.x) / depth + cx
    pixel_y = fy * float(colmap_position.y) / depth + cy
    if not (0.0 <= pixel_x < width and 0.0 <= pixel_y < height):
        return None
    return float(pixel_x), float(pixel_y)


def _projected_segment_may_touch_image(first, second, width, height):
    """Return a conservative screen-space intersection test for an edge."""
    return not (
        max(first[0], second[0]) < 0.0 or
        min(first[0], second[0]) >= width or
        max(first[1], second[1]) < 0.0 or
        min(first[1], second[1]) >= height
    )


def _border_point_is_visible(frame, position, tree):
    """Check a border point against opaque geometry in the ray BVH."""
    pixel = frame.project(position)
    if pixel is None:
        return None
    hit, _, _, _ = frame.cast(tree, pixel[0], pixel[1])
    if hit is None:
        return pixel
    hit_projection = frame.project_unbounded(hit)
    if hit_projection is None:
        return pixel
    # Allow an opaque frame or trim that occupies the same surface as the
    # transparent pane, but reject opaque geometry in front of the border.
    epsilon = max(1e-6, pixel[2] * 2e-6)
    return pixel if hit_projection[2] + epsilon >= pixel[2] else None


def _add_glass_border_points(edges, frames, points, tree, pixel_spacing):
    """Sample visible transparent border edges into a PointBuffer."""
    border_indices = set()
    spacing = max(1, int(pixel_spacing))
    for first, second in edges:
        projections = []
        for frame in frames:
            start = frame.project_unbounded(first)
            end = frame.project_unbounded(second)
            if start is None or end is None:
                continue
            if _projected_segment_may_touch_image(start, end, frame.width, frame.height):
                length = math.hypot(end[0] - start[0], end[1] - start[1])
                projections.append(length)
        if not projections:
            continue

        subdivisions = max(1, math.ceil(max(projections) / spacing))
        start = Vector(first)
        delta = Vector(second) - start
        for step in range(subdivisions + 1):
            position = start + delta * (step / subdivisions)
            if not any(_border_point_is_visible(frame, position, tree) is not None
                       for frame in frames):
                continue
            candidate = points.find(position, GLASS_BORDER_TRIANGLE)
            if candidate is None:
                candidate = points.add(tuple(position), GLASS_BORDER_TRIANGLE)
            border_indices.add(candidate)
    return sorted(border_indices)


def _link_glass_border_points(edges, frames, images, images_dir, points, tree, pixel_spacing):
    """Add image tracks and saved-image colors for sampled glass borders."""
    border_indices = _add_glass_border_points(edges, frames, points, tree, pixel_spacing)
    if not border_indices:
        return 0

    for image_index, (frame, image) in enumerate(zip(frames, images)):
        observations = image["points2d"]
        with ImagePixels(images_dir / image["name"], frame.width, frame.height) as pixels:
            fx, fy, _, _ = frame.params
            for point_index in border_indices:
                pixel = _border_point_is_visible(frame, points.position(point_index), tree)
                if pixel is None:
                    continue
                point2d_index = len(observations)
                footprint = pixel[2] ** 2 / (fx * fy)
                if points.observe(
                        point_index,
                        image_index + 1,
                        point2d_index,
                        pixels.rgb(*pixel[:2]),
                        footprint,
                ):
                    observations.append(pixel[0], pixel[1], point_index + 1)
    return len(border_indices)


def _link_mesh_points_to_images(points, perspective_cameras, images, scene, width, height):
    """Create image observations and COLMAP tracks for mesh-derived points.

    A point is retained only when it projects in front of and inside at least
    one exported camera.  Occlusion is not inferred here: the source geometry
    is already known, and the generated observations are exact projections of
    the exported camera poses.
    """
    image_observations = [[] for _ in images]
    point_tracks = [[] for _ in points]
    observed_point_indices = set()

    for point_index, point in enumerate(points):
        for image_index, camera_object in enumerate(perspective_cameras):
            pixel = _project_mesh_point(
                camera_object,
                point["position"],
                camera_object.data,
                scene,
                width,
                height,
            )
            if pixel is None:
                continue
            point2d_index = len(image_observations[image_index])
            image_observations[image_index].append((pixel[0], pixel[1], point_index))
            point_tracks[point_index].append((image_index + 1, point2d_index))
            observed_point_indices.add(point_index)

    point_id_by_index = {
        point_index: point_id
        for point_id, point_index in enumerate(sorted(observed_point_indices), start=1)
    }
    for image, observations in zip(images, image_observations):
        image["points2d"] = [
            (pixel_x, pixel_y, point_id_by_index[point_index])
            for pixel_x, pixel_y, point_index in observations
            if point_index in point_id_by_index
        ]

    linked_points = []
    for point_index in sorted(observed_point_indices):
        point = dict(points[point_index])
        point["tracks"] = point_tracks[point_index]
        linked_points.append(point)
    return linked_points


def _write_colmap_text_steps(output_dir, cameras, images, points=None):
    """Stream COLMAP text, yielding between chunks for UI cancellation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    points = points or []

    with (output_dir / "cameras.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id, camera in cameras:
            fx, fy, cx, cy = camera["params"]
            handle.write(
                f"{camera_id} PINHOLE {camera['width']} {camera['height']} "
                f"{fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n"
            )

    with (output_dir / "images.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        observation_count = sum(len(image.get("points2d", ())) for _, image in images)
        mean_observations = observation_count / len(images) if images else 0.0
        handle.write(
            f"# Number of images: {len(images)}, "
            f"mean observations per image: {mean_observations:.12g}\n"
        )
        for image_id, image in images:
            pose = " ".join(f"{value:.12g}" for value in image["pose"])
            handle.write(
                f"{image_id} {pose} {image_id} {image['name']}\n"
            )
            points2d = image.get("points2d", ())
            chunk = []
            first = True
            for pixel_x, pixel_y, point_id in points2d:
                chunk.append(f"{pixel_x:.12g} {pixel_y:.12g} {point_id}")
                if len(chunk) == 4096:
                    handle.write(("" if first else " ") + " ".join(chunk))
                    first = False
                    chunk.clear()
                    yield
            if chunk:
                handle.write(("" if first else " ") + " ".join(chunk))
            handle.write("\n")
            yield

    with (output_dir / "points3D.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        track_count = (len(points.track_image) if isinstance(points, PointBuffer)
                       else sum(len(point.get("tracks", ())) for point in points))
        mean_track_length = track_count / len(points) if points else 0.0
        handle.write(
            f"# Number of points: {len(points)}, "
            f"mean track length: {mean_track_length:.12g}\n"
        )
        for point_id, point in enumerate(points, start=1):
            x, y, z = point["position"]
            red, green, blue = point["color"]
            tracks = " ".join(
                f"{image_id} {point2d_index}"
                for image_id, point2d_index in point.get("tracks", ())
            )
            handle.write(
                f"{point_id} {x:.12g} {y:.12g} {z:.12g} "
                f"{red} {green} {blue} 0"
                f"{(' ' + tracks) if tracks else ''}\n"
            )
            if point_id % 4096 == 0:
                yield


def _write_colmap_text(output_dir, cameras, images, points=None):
    for _ in _write_colmap_text_steps(output_dir, cameras, images, points):
        pass


class GS_ColmapExportSettings(bpy.types.PropertyGroup):
    output_path: StringProperty(
        name="Output directory",
        description="Directory where the COLMAP model and rendered images are saved",
        default="//colmap_export",
        subtype="DIR_PATH",
    )
    render_images: BoolProperty(
        name="Render images",
        description="Render one image from every perspective camera in the scene",
        default=True,
    )
    skip_existing_images: BoolProperty(
        name="Skip existing images",
        description="Reuse decodable images with matching export dimensions from the images folder",
        default=True,
    )
    render_format: EnumProperty(
        name="Image format",
        description="Format for images rendered for the COLMAP dataset",
        items=[
            ("PNG", "PNG", "Lossless image output"),
            ("JPEG", "JPEG", "Smaller lossy image output"),
        ],
        default="PNG",
    )
    export_mesh_points: BoolProperty(
        name="Export Blender points",
        description="Write Blender-derived surface points to points3D.txt",
        default=True,
    )
    point_source: EnumProperty(
        name="Point source",
        items=[("RAYS", "Camera rays", "Sample visible surfaces using every camera and saved image colors"),
               ("VERTICES", "Mesh vertices", "Legacy original vertices and material colors; no occlusion test")],
        default="RAYS",
    )
    pixel_spacing: IntProperty(
        name="Pixel spacing", description="Ray spacing in pixels; 1 samples every pixel without a point cap",
        default=1, min=1, soft_max=32,
    )
    merge_distance: FloatProperty(
        name="Merge distance", description="Merge hits on the same instance triangle within this distance in Blender units",
        default=0.000001, min=0.000000001, soft_max=0.01, precision=6,
    )
    skip_transparent: BoolProperty(
        name="Pass through glass",
        description="Skip detected glass, transmission, and transparent faces so rays reach geometry behind them",
        default=True,
    )
    include_glass_borders: BoolProperty(
        name="Include glass borders",
        description="Add sampled boundary points for transparent/glass faces while passing rays through their interiors",
        default=True,
    )


class GS_OT_export_colmap(bpy.types.Operator):
    bl_idname = "gs_camera_rig.export_colmap"
    bl_label = "Export COLMAP Dataset"
    bl_description = "Export every perspective camera in the scene as a COLMAP model"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return active_export is None

    def _export_steps(self, context):
        scene = context.scene
        settings = scene.gs_colmap_export
        use_points = settings.export_mesh_points
        ray_mode = use_points and settings.point_source == "RAYS"
        render_images = settings.render_images
        skip_existing = settings.skip_existing_images
        render_format = settings.render_format
        spacing = settings.pixel_spacing
        merge_distance = settings.merge_distance
        include_glass_borders = settings.include_glass_borders
        cameras = _scene_camera_objects(scene)
        perspective_cameras = [obj for obj in cameras if obj.data.type == "PERSP"]
        skipped = len(cameras) - len(perspective_cameras)
        if not perspective_cameras:
            raise ValueError("No perspective cameras found in the scene")
        output_dir = Path(bpy.path.abspath(settings.output_path)).resolve()
        images_dir = output_dir / "images"
        width, height = _render_dimensions(scene)
        extension = "jpg" if render_format == "JPEG" else "png"
        depsgraph = context.evaluated_depsgraph_get()
        frames = [CameraFrame(camera, depsgraph, scene, width, height) for camera in perspective_cameras]
        colmap_cameras = []
        colmap_images = []
        used_names = set()
        for camera_object, frame in zip(perspective_cameras, frames):
            image_name = _safe_image_name(camera_object, extension)
            stem = Path(image_name).stem
            suffix = 2
            while image_name.casefold() in used_names:
                image_name = f"{stem}_{suffix}.{extension}"
                suffix += 1
            used_names.add(image_name.casefold())
            colmap_cameras.append(
                {
                    "width": width,
                    "height": height,
                    "params": frame.params,
                }
            )
            colmap_images.append(
                {
                    "name": image_name,
                    "pose": frame.pose,
                }
            )

        original_camera = scene.camera
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        original_extension = scene.render.use_file_extension
        original_use_border = scene.render.use_border
        original_use_crop_to_border = getattr(scene.render, "use_crop_to_border", None)
        original_use_multiview = scene.render.use_multiview
        rendered_count = 0
        reused_count = 0
        started = time.perf_counter()
        ray_count = len(frames) * math.ceil(width / spacing) * math.ceil(height / spacing)
        self.report({"INFO"}, f"Sampling up to {ray_count:,} camera rays (no point cap)" if ray_mode
                    else "Preparing COLMAP export")
        try:
            # COLMAP PINHOLE intrinsics describe the complete saved image. A
            # border render or multiview render changes the pixel domain, so
            # turn those options off for this operation and restore them in
            # the finally block below. This keeps the user's scene settings
            # intact while making the export deterministic.
            if original_use_border or original_use_multiview:
                self.report({"WARNING"}, "Temporarily disabling render border/cropping and multiview for full-frame COLMAP images")
            scene.render.use_border = False
            if original_use_crop_to_border is not None:
                scene.render.use_crop_to_border = False
            scene.render.use_multiview = False
            output_dir.mkdir(parents=True, exist_ok=True)
            # Validate every reused image before doing expensive geometry work.
            valid_images = set()
            for index, image in enumerate(colmap_images):
                image_path = images_dir / image["name"]
                if ray_mode or (render_images and skip_existing):
                    if not render_images or skip_existing:
                        try:
                            with ImagePixels(image_path, width, height):
                                valid_images.add(index)
                        except Exception as exc:
                            if not render_images:
                                raise ValueError(f"{exc}. Enable Render images or supply a matching image") from exc
                            if image_path.exists():
                                self.report({"WARNING"}, f"Re-rendering invalid image '{image['name']}': {exc}")
                yield 5 * (index + 1) / len(frames), f"Checking images {index + 1}/{len(frames)}"
            if render_images:
                images_dir.mkdir(parents=True, exist_ok=True)
                scene.render.image_settings.file_format = render_format
                scene.render.use_file_extension = True
                for index, (camera_object, image) in enumerate(zip(perspective_cameras, colmap_images)):
                    image_path = images_dir / image["name"]
                    if index in valid_images:
                        reused_count += 1
                    else:
                        yield 5 + 25 * index / len(frames), f"Rendering {camera_object.name} (Esc cancels render)"
                        scene.camera = camera_object
                        scene.render.filepath = str(image_path)
                        result = bpy.ops.render.render(write_still=True)
                        if "CANCELLED" in result:
                            raise RuntimeError("Image rendering was cancelled")
                        with ImagePixels(image_path, width, height):
                            pass
                        rendered_count += 1
                    yield 5 + 25 * (index + 1) / len(frames), f"Images {index + 1}/{len(frames)} ready"
            else:
                reused_count = len(valid_images)

            points = []
            if ray_mode:
                yield 30, "Building geometry index"
                tree, warnings, triangle_count = build_geometry(
                    context, skip_transparent=settings.skip_transparent,
                )
                for message in warnings:
                    print(f"COLMAP warning: {message}")
                if warnings:
                    self.report({"WARNING"}, f"{len(warnings)} geometry warning(s); see console. {warnings[0]}")
                if any(camera.data.dof.use_dof for camera in perspective_cameras) or getattr(scene.render, "use_motion_blur", False):
                    self.report({"WARNING"}, "Rays use a static pinhole camera; rendered depth of field/motion blur may differ")
                points = PointBuffer(merge_distance)
                per_camera = math.ceil(width / spacing) * math.ceil(height / spacing)
                try:
                    for index, (frame, image) in enumerate(zip(frames, colmap_images)):
                        observations = Observations()
                        image["points2d"] = observations
                        with ImagePixels(images_dir / image["name"], width, height) as pixels:
                            for completed in sample_camera(tree, frame, pixels, points, observations,
                                                           index + 1, spacing):
                                done = index * per_camera + completed
                                yield 35 + 55 * done / ray_count, f"Rays {done:,}/{ray_count:,}; points {len(points):,}"
                    if include_glass_borders:
                        yield 91, "Adding glass border points"
                        border_edges = collect_transparent_border_edges(context)
                        border_count = _link_glass_border_points(
                            border_edges,
                            frames,
                            colmap_images,
                            images_dir,
                            points,
                            tree,
                            spacing,
                        )
                        if border_count:
                            print(f"COLMAP: added {border_count:,} glass border point(s)")
                finally:
                    tree = None
                    points.release_spatial_index()
                print(f"COLMAP: {triangle_count:,} evaluated triangles, {ray_count:,} rays, {len(points):,} points")
            elif use_points:
                yield 35, "Collecting legacy mesh vertices"
                points = _link_mesh_points_to_images(
                    _collect_mesh_points(scene), perspective_cameras, colmap_images, scene, width, height,
                )

            # TemporaryDirectory cleans up uncommitted model files on Esc/errors.
            with tempfile.TemporaryDirectory(prefix=".gs-colmap-", dir=output_dir) as temporary:
                stage = Path(temporary)
                for _ in _write_colmap_text_steps(stage, list(enumerate(colmap_cameras, start=1)),
                                                 list(enumerate(colmap_images, start=1)), points):
                    yield 95, f"Writing model: {len(points):,} points"
                yield 99, "Committing COLMAP model"
                commit_model(stage, output_dir)
            elapsed = time.perf_counter() - started
            message = (f"Exported {len(frames)} cameras and {len(points):,} points in {elapsed:.1f}s "
                       f"({rendered_count} rendered, {reused_count} reused) to {output_dir}")
            if skipped:
                message += f"; skipped {skipped} non-perspective camera(s)"
            return message
        finally:
            scene.camera = original_camera
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            scene.render.use_file_extension = original_extension
            scene.render.use_border = original_use_border
            if original_use_crop_to_border is not None:
                scene.render.use_crop_to_border = original_use_crop_to_border
            scene.render.use_multiview = original_use_multiview

    def _start(self, context):
        global active_export
        active_export = self
        self._timer = None
        self._steps = self._export_steps(context)
        self._window_manager = context.window_manager
        self._workspace = context.workspace
        self._window_manager.progress_begin(0, 100)

    def _cleanup(self):
        global active_export
        try:
            if getattr(self, "_steps", None) is not None:
                self._steps.close()
                self._steps = None
        finally:
            if getattr(self, "_timer", None) is not None:
                self._window_manager.event_timer_remove(self._timer)
                self._timer = None
            self._window_manager.progress_end()
            if self._workspace:
                self._workspace.status_text_set(None)
            active_export = None

    def _advance(self):
        try:
            progress, message = next(self._steps)
            self._window_manager.progress_update(progress)
            if self._workspace:
                self._workspace.status_text_set(f"COLMAP: {message} — Esc to cancel")
            return {"RUNNING_MODAL"}
        except StopIteration as done:
            self._cleanup()
            self.report({"INFO"}, done.value or "COLMAP export completed")
            return {"FINISHED"}
        except Exception as exc:
            self._cleanup()
            self.report({"ERROR"}, f"COLMAP export failed: {exc}")
            return {"CANCELLED"}

    def execute(self, context):
        self._start(context)
        while True:
            result = self._advance()
            if result != {"RUNNING_MODAL"}:
                return result

    def invoke(self, context, event):
        if bpy.app.background or context.window is None:
            return self.execute(context)
        self._start(context)
        self._timer = context.window_manager.event_timer_add(0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup()
            self.report({"INFO"}, "COLMAP export cancelled; model files were not replaced")
            return {"CANCELLED"}
        if _is_export_timer_event(event, self._timer):
            return self._advance()
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._cleanup()


def _ensure_collection(scene):
    """Return the dedicated camera collection, creating it if necessary."""
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION_NAME)
    if collection.name not in {child.name for child in scene.collection.children}:
        scene.collection.children.link(collection)
    return collection


def _view_basis(context):
    """Return (position, front, right, up, distance) for the active 3D view.

    ``front`` points from the viewer into the scene.  The other vectors are
    all world-space and form a right-handed basis around the current view.
    """
    area = context.area if context.area and context.area.type == "VIEW_3D" else None
    space = area.spaces.active if area else None
    region_3d = space.region_3d if space else None

    if region_3d is None:
        # Operators can also be called from a non-viewport context.  The
        # fallback keeps the operator useful from Python and the F3 menu.
        position = context.scene.cursor.location.copy()
        front = Vector((0.0, 0.0, -1.0))
        right = Vector((1.0, 0.0, 0.0))
        up = Vector((0.0, 1.0, 0.0))
        return position, front, right, up, 5.0

    rotation = region_3d.view_rotation.copy()
    front = (rotation @ Vector((0.0, 0.0, -1.0))).normalized()
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    distance = max(float(region_3d.view_distance), 0.01)
    # The inverse view matrix gives the exact current viewport/camera
    # position, including camera-view mode and any viewport navigation.
    position = region_3d.view_matrix.inverted().translation.copy()
    return position, front, right, up, distance


def _view_rotation(context):
    """Return the current viewport rotation when the operator has a 3D view."""
    area = context.area if context.area and context.area.type == "VIEW_3D" else None
    space = area.spaces.active if area else None
    region_3d = space.region_3d if space else None
    return region_3d.view_rotation.copy() if region_3d else None


def _yaw_camera_rotation(rotation, angle):
    """Rotate a camera pose around world Z while preserving its pitch.

    Applying the yaw to the complete pose keeps every generated camera
    looking down by the same amount, including the exact straight-down case
    where the forward vector alone would not show any yaw difference.
    """
    return Matrix.Rotation(angle, 4, "Z") @ rotation.to_matrix().to_4x4()


def _point_camera(obj, direction):
    """Aim a Blender camera (whose forward axis is local -Z) at a direction."""
    direction = direction.normalized()
    if direction.length == 0:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _ring_directions(count, is_arc):
    """Directions in the local (right, front, up) frame for a horizontal orbit."""
    if count == 1:
        angles = [0.0]
    elif is_arc:
        # Start at the current view, then alternate left/right until the
        # complete front-facing half-circle has been covered.
        steps = math.ceil((count - 1) / 2)
        step = (math.pi / 2.0) / steps
        angles = [0.0]
        for ring_index in range(1, steps + 1):
            for side in (1.0, -1.0):
                if len(angles) >= count:
                    break
                angles.append(side * ring_index * step)
    else:
        angles = [2.0 * math.pi * i / count for i in range(count)]

    # theta=0 is in front of the target, i.e. opposite the view direction.
    # Local coordinates are (right, forward, up). theta=0 is the current
    # camera direction, so camera 1 starts from the user's current rotation.
    return [Vector((math.sin(theta), math.cos(theta), 0.0)) for theta in angles]


def _sphere_directions(count, preset):
    """Fibonacci directions with optional directional exclusions.

    The local frame is (right, front, up), where front is the direction from
    the target toward the viewer.  Candidate oversampling makes the requested
    camera count stable even when a preset excludes a region of the sphere.
    """
    accepted = []
    candidate_count = max(count * 8, 32)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for index in range(candidate_count):
        y = 1.0 - 2.0 * (index + 0.5) / candidate_count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        angle = golden_angle * index
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius

        # x = right, y = up, z = forward/back around the current view.
        right = x
        up = y
        front = z

        if preset == "SPHERE_NO_TOP" and up > 0.25:
            continue
        if preset == "SPHERE_NO_LEFT" and right < -0.25:
            continue
        if preset == "SPHERE_NO_RIGHT" and right > 0.25:
            continue
        if preset == "SPHERE_NO_LEFT_RIGHT" and abs(right) > 0.25:
            continue
        accepted.append(Vector((right, front, up)))

    # If a very small count is requested, or a future preset gets stricter,
    # retain a deterministic fallback rather than silently creating too few.
    if len(accepted) < count:
        accepted = [Vector((0.0, 1.0, 0.0))] * count
    return accepted[:count]


def _grid_values(start, end, count):
    """Return evenly spaced grid offsets, including both endpoints."""
    if count <= 0:
        return []
    if count <= 1:
        return [(start + end) * 0.5]
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def _grid_dimensions(camera_count):
    """Return a compact rectangular layout for one total camera count."""
    if camera_count <= 0:
        return 0, 0
    x_count = max(1, math.ceil(math.sqrt(camera_count)))
    y_count = max(1, math.ceil(camera_count / x_count))
    return x_count, y_count


def _grid_offsets(settings, view_distance=1.0):
    """Return centered X/Y spawn offsets for the camera array."""
    x_count, y_count = _grid_dimensions(settings.camera_count)
    if not x_count or not y_count:
        return []

    x_span = view_distance * GRID_CAMERA_SPAN_SCALE * settings.grid_x_range / (2.0 * math.pi)
    y_span = view_distance * GRID_CAMERA_SPAN_SCALE * settings.grid_y_range / (2.0 * math.pi)
    x_offsets = _grid_values(-x_span * 0.5, x_span * 0.5, x_count)
    y_offsets = _grid_values(-y_span * 0.5, y_span * 0.5, y_count)
    offsets = []
    for y_offset in y_offsets:
        for x_offset in x_offsets:
            if len(offsets) >= settings.camera_count:
                return offsets
            offsets.append((x_offset, y_offset))
    return offsets


def _grid_directions(settings):
    """Return one shared current-view direction for every grid camera."""
    return [Vector((0.0, 1.0, 0.0)) for _ in _grid_offsets(settings)]


def _directions(settings):
    if settings.preset == "RING_360":
        return _ring_directions(settings.camera_count, False)
    if settings.preset == "ARC_180":
        return _ring_directions(settings.camera_count, True)
    if settings.preset == "GRID_2D":
        return _grid_directions(settings)
    directions = _sphere_directions(settings.camera_count, settings.preset)
    # A spherical layout also begins with a camera at the user's current
    # position; the remaining cameras keep the Fibonacci distribution.
    directions[0] = Vector((0.0, 1.0, 0.0))
    return directions


def _local_to_world(local_direction, right, front, up):
    return (right * local_direction.x + front * local_direction.y + up * local_direction.z).normalized()


def _unique_camera_name(prefix, index):
    base = f"{prefix}{index:03d}"
    if bpy.data.objects.get(base) is None:
        return base
    suffix = 1
    while bpy.data.objects.get(f"{base}_{suffix}") is not None:
        suffix += 1
    return f"{base}_{suffix}"


class GS_CameraRigSettings(bpy.types.PropertyGroup):
    camera_count: IntProperty(
        name="Camera count",
        description="Number of cameras to generate",
        default=24,
        min=1,
        max=512,
    )
    preset: EnumProperty(
        name="Preset",
        description="How cameras are distributed around the viewed point",
        items=[
            ("RING_360", "360° horizontal ring", "Evenly spaced around the full horizon"),
            ("ARC_180", "180° front arc", "Evenly spaced across the half-circle in front of the current view"),
            ("GRID_2D", "2D camera grid", "Spread cameras across view-relative horizontal and vertical ranges"),
            ("SPHERE_360", "360° sphere", "Fibonacci distribution around the full sphere"),
            ("SPHERE_NO_TOP", "Sphere — ignore top", "Full sphere with the upper region removed"),
            ("SPHERE_NO_LEFT", "Sphere — ignore left", "Full sphere with the left region removed"),
            ("SPHERE_NO_RIGHT", "Sphere — ignore right", "Full sphere with the right region removed"),
            ("SPHERE_NO_LEFT_RIGHT", "Sphere — ignore left + right", "Full sphere with both side regions removed"),
        ],
        default="RING_360",
    )
    grid_x_range: FloatProperty(
        name="X spawn range",
        description="Horizontal angular range used to spread the camera array",
        default=math.radians(90.0),
        min=0.0,
        max=2.0 * math.pi,
        subtype="ANGLE",
    )
    grid_y_range: FloatProperty(
        name="Y spawn range",
        description="Vertical angular range used to spread the camera array",
        default=math.radians(60.0),
        min=0.0,
        max=2.0 * math.pi,
        subtype="ANGLE",
    )
    lens: FloatProperty(
        name="Lens",
        description="Focal length applied to every generated camera",
        default=20.0,
        min=1.0,
        max=250.0,
        unit="LENGTH",
    )
    prefix: StringProperty(
        name="Name prefix",
        description="Prefix for generated camera object names",
        default="GS_Cam_",
    )
    make_active: BoolProperty(
        name="Make first camera active",
        description="Set the first generated camera as the scene camera",
        default=False,
    )
_listener_options = {"INTERNAL"}
if bpy.app.version >= (4, 2, 0):
    # Blender 4.2 added modal-priority operators, which lets this listener
    # see the shortcut before Walk/Fly while passing all other events through.
    _listener_options.add("MODAL_PRIORITY")


class GS_OT_capture_hotkey_listener(bpy.types.Operator):
    bl_idname = "gs_camera_rig.capture_hotkey_listener"
    bl_label = "Camera Rig Hotkey Listener"
    bl_description = "Keep the camera-generation shortcut available during Walk/Fly navigation"
    bl_options = _listener_options

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if listener_shutdown:
            return {"FINISHED"}
        if (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and (_is_spawn_hotkey(event) or (_is_walk_fly_active(context) and _is_walk_fly_alias(event)))
        ):
            # Consume only this shortcut. Every other event is passed through
            # so native Walk/Fly navigation remains unchanged.
            bpy.ops.gs_camera_rig.generate_cameras("EXEC_DEFAULT")
            return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL", "PASS_THROUGH"}


def _start_hotkey_listeners():
    """Start one pass-through listener in each available 3D View window."""
    if listener_shutdown:
        return None

    window_manager = bpy.context.window_manager
    if window_manager is None:
        return 0.5

    for window in window_manager.windows:
        try:
            window_id = window.as_pointer()
            if window_id in listener_windows:
                continue
            area = next((area for area in window.screen.areas if area.type == "VIEW_3D"), None)
            if area is None:
                continue
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                result = bpy.ops.gs_camera_rig.capture_hotkey_listener("INVOKE_DEFAULT")
            if "RUNNING_MODAL" in result:
                listener_windows.add(window_id)
        except Exception as error:
            # Timer callbacks are removed by Blender when they raise. Keep
            # retrying so a temporary context transition cannot disable the
            # Walk/Fly shortcut permanently.
            print(f"Gaussian Camera Rig: listener startup retry: {error}")

    return 0.5


class GS_OT_generate_cameras(bpy.types.Operator):
    bl_idname = "gs_camera_rig.generate_cameras"
    bl_label = "Generate Cameras"
    bl_description = "Create cameras using the current viewport direction"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.gs_camera_rig
        position, view_front, view_right, view_up, view_distance = _view_basis(context)
        current_rotation = _view_rotation(context)
        prefix = settings.prefix or "GS_Cam_"
        collection = _ensure_collection(context.scene)
        directions = _directions(settings)
        grid_offsets = _grid_offsets(settings, view_distance) if settings.preset == "GRID_2D" else None
        generated_names = []

        for index, local_direction in enumerate(directions, start=1):
            camera_data = bpy.data.cameras.new(f"{prefix}{index:03d}_DATA")
            # Deliberately use an independent, identical framing for the
            # entire generated batch; never copy the current camera lens.
            camera_data.type = "PERSP"
            camera_data.lens_unit = "MILLIMETERS"
            camera_data.lens = settings.lens
            camera_data.display_size = max(view_distance * 0.08, 0.05)

            name = _unique_camera_name(prefix, index)
            camera_object = bpy.data.objects.new(name, camera_data)
            collection.objects.link(camera_object)

            if settings.preset == "GRID_2D":
                x_offset, y_offset = grid_offsets[index - 1]
                camera_object.location = (
                    position + view_right * x_offset + view_up * y_offset
                )
                if current_rotation is not None:
                    # Every array camera has the exact orientation of the
                    # viewport/camera the user is currently looking through.
                    camera_object.rotation_euler = current_rotation.to_euler()
                else:
                    _point_camera(camera_object, view_front)
            elif current_rotation is not None and settings.preset in {"RING_360", "ARC_180"}:
                camera_object.location = position
                # Horizontal presets change heading only. Applying the yaw
                # to the complete current pose preserves a downward-looking
                # viewport orientation for every camera in the batch.
                angle = math.atan2(local_direction.x, local_direction.y)
                camera_object.rotation_euler = _yaw_camera_rotation(
                    current_rotation, angle
                ).to_euler()
            else:
                world_direction = _local_to_world(local_direction, view_right, view_front, view_up)
                camera_object.location = position
                _point_camera(camera_object, world_direction)
            generated_names.append(camera_object.name)

        # ID properties are consistently string/number based across Blender
        # 3.6 through 4.x; keep the batch as a newline-separated string.
        context.scene[LAST_BATCH_KEY] = "\n".join(generated_names)
        if settings.make_active and generated_names:
            context.scene.camera = bpy.data.objects.get(generated_names[0])

        if settings.preset == "GRID_2D":
            message = (
                f"Generated {len(generated_names)} cameras in a "
                f"{_grid_dimensions(settings.camera_count)[0]} × "
                f"{_grid_dimensions(settings.camera_count)[1]} array "
                f"with the current looking direction."
            )
        else:
            message = f"Generated {len(generated_names)} fixed-position cameras at {position[:]}."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class GS_OT_undo_last_batch(bpy.types.Operator):
    bl_idname = "gs_camera_rig.undo_last_batch"
    bl_label = "Undo Last Camera Batch"
    bl_description = "Delete the cameras created by the last generation (Ctrl+Z also undoes generation)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        batch_value = context.scene.get(LAST_BATCH_KEY, "")
        names = [name for name in str(batch_value).splitlines() if name]
        removed = 0
        for name in names:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "CAMERA":
                continue
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        context.scene[LAST_BATCH_KEY] = ""
        self.report({"INFO"}, f"Removed {removed} generated cameras.")
        return {"FINISHED"}


class GS_PT_camera_rig(bpy.types.Panel):
    bl_label = "Gaussian Cameras"
    bl_idname = "GS_PT_camera_rig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gaussian Cameras"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.gs_camera_rig

        box = layout.box()
        title = "Generate camera grid" if settings.preset == "GRID_2D" else "Generate fixed-position cameras"
        box.label(text=title, icon="CAMERA_DATA")
        box.prop(settings, "preset")
        box.prop(settings, "camera_count")
        grid_box = box.box()
        grid_box.label(text="3D camera spawn range")
        row = grid_box.row(align=True)
        row.prop(settings, "grid_x_range", slider=True)
        row.prop(settings, "grid_y_range", slider=True)
        grid_box.label(
            text=f"{_grid_dimensions(settings.camera_count)[0]} × "
                f"{_grid_dimensions(settings.camera_count)[1]} layout from "
                f"{settings.camera_count} camera(s)"
        )
        grid_box.label(text="X and Y spawn ranges: 0° to 360°.")

        box.prop(settings, "lens")
        box.prop(settings, "prefix")
        box.prop(settings, "make_active")

        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator(GS_OT_generate_cameras.bl_idname, icon="ADD")

        last_batch = context.scene.get(LAST_BATCH_KEY, "")
        row = layout.row(align=True)
        row.enabled = bool(last_batch)
        row.operator(GS_OT_undo_last_batch.bl_idname, icon="LOOP_BACK")
        if settings.preset == "GRID_2D":
            layout.label(text="Grid cameras share the current looking direction.", icon="INFO")
        else:
            layout.label(text="All cameras share your current position.", icon="INFO")
        layout.label(text="Ctrl+Z undoes the latest generation.", icon="INFO")


class GS_PT_colmap_export(bpy.types.Panel):
    bl_label = "COLMAP Export"
    bl_idname = "GS_PT_colmap_export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gaussian Cameras"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.gs_colmap_export
        cameras = _scene_camera_objects(scene)
        perspective_count = sum(obj.data.type == "PERSP" for obj in cameras)

        box = layout.box()
        box.label(text=f"Scene cameras: {len(cameras)}", icon="CAMERA_DATA")
        box.label(text=f"Perspective cameras: {perspective_count}", icon="VIEW_CAMERA")
        box.label(text="Export includes every scene camera.", icon="INFO")

        layout.prop(settings, "output_path")
        layout.prop(settings, "render_images")
        layout.prop(settings, "render_format")
        if settings.render_images:
            layout.prop(settings, "skip_existing_images")
        layout.prop(settings, "export_mesh_points")
        if settings.export_mesh_points:
            layout.prop(settings, "point_source")
            if settings.point_source == "RAYS":
                layout.prop(settings, "pixel_spacing")
                layout.prop(settings, "merge_distance")
                layout.prop(settings, "skip_transparent")
                layout.prop(settings, "include_glass_borders")
                width, height = _render_dimensions(scene)
                rays = perspective_count * math.ceil(width / settings.pixel_spacing) * math.ceil(height / settings.pixel_spacing)
                layout.label(text=f"Estimated rays: {rays:,} (no point cap)")
                layout.label(text="Colors require matching images.", icon="IMAGE_DATA")

        row = layout.row()
        row.scale_y = 1.5
        row.enabled = perspective_count > 0 and active_export is None
        row.operator(GS_OT_export_colmap.bl_idname, icon="EXPORT")

        layout.label(text="Writes cameras.txt, images.txt, points3D.txt.", icon="FILE_TEXT")
        layout.label(text="Camera rays sample the first mesh surface.", icon="INFO")

classes = (
    GS_CameraRigSettings,
    GS_OT_capture_hotkey_listener,
    GS_OT_generate_cameras,
    GS_OT_undo_last_batch,
    GS_PT_camera_rig,
    GS_ColmapExportSettings,
    GS_OT_export_colmap,
    GS_PT_colmap_export,
)


def register():
    global listener_shutdown
    listener_shutdown = False
    listener_windows.clear()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gs_camera_rig = PointerProperty(type=GS_CameraRigSettings)
    bpy.types.Scene.gs_colmap_export = PointerProperty(type=GS_ColmapExportSettings)

    # Register the shortcut in the 3D View only, so it does not interfere
    # with text fields or shortcuts in other Blender editors.
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:
        keymap = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
        keymap_item = keymap.keymap_items.new(
            GS_OT_generate_cameras.bl_idname,
            type="C",
            value="PRESS",
            ctrl=True,
            shift=True,
            alt=True,
        )
        addon_keymaps.append((keymap, keymap_item))
    bpy.app.timers.register(_start_hotkey_listeners, first_interval=0.2)


def unregister():
    global listener_shutdown
    if active_export is not None:
        active_export._cleanup()
    listener_shutdown = True
    if bpy.app.timers.is_registered(_start_hotkey_listeners):
        bpy.app.timers.unregister(_start_hotkey_listeners)
    listener_windows.clear()
    for keymap, keymap_item in addon_keymaps:
        keymap.keymap_items.remove(keymap_item)
    addon_keymaps.clear()
    if hasattr(bpy.types.Scene, "gs_colmap_export"):
        del bpy.types.Scene.gs_colmap_export
    if hasattr(bpy.types.Scene, "gs_camera_rig"):
        del bpy.types.Scene.gs_camera_rig
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
