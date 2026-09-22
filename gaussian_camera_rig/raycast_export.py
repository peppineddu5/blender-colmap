"""Camera-ray sampling and compact COLMAP buffers; only Blender/stdlib required."""

from array import array
from contextlib import contextmanager
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


class CameraFrame:
    """One immutable calibration shared by projection, ray casting and export."""

    def __init__(self, camera, depsgraph, scene, width, height):
        evaluated = camera.evaluated_get(depsgraph)
        projection = evaluated.calc_matrix_camera(
            depsgraph, x=width, y=height,
            scale_x=scene.render.pixel_aspect_x,
            scale_y=scene.render.pixel_aspect_y,
        )
        self.params = (
            projection[0][0] * width / 2,
            projection[1][1] * height / 2,
            (1 - projection[0][2]) * width / 2,
            (1 + projection[1][2]) * height / 2,
        )
        self.width, self.height = width, height
        world = evaluated.matrix_world.copy()
        self.center = world.translation.copy()
        # Blender ignores camera scale, including scale inherited from a parent.
        self.rotation = world.to_3x3().normalized()
        identity = self.rotation.transposed() @ self.rotation
        if (self.rotation.determinant() < 0.9999 or
                any(abs(identity[i][j] - (i == j)) > 1e-4
                    for i in range(3) for j in range(3))):
            raise ValueError(f"Camera '{camera.name}' has shear, reflection or zero scale; apply its transforms")
        self.inverse_rotation = self.rotation.transposed()
        self.near = float(evaluated.data.clip_start)
        self.far = float(evaluated.data.clip_end)
        conversion = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        rotation = conversion @ self.inverse_rotation
        translation = -(rotation @ self.center)
        self.pose = tuple(rotation.to_quaternion()) + tuple(translation)

    def project(self, position):
        projection = self.project_unbounded(position)
        if projection is None:
            return None
        x, y, depth = projection
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        return x, y, depth

    def project_unbounded(self, position):
        """Project a point without rejecting pixels outside the image."""
        local = self.inverse_rotation @ (Vector(position) - self.center)
        depth = -local.z
        if not self.near <= depth <= self.far:
            return None
        fx, fy, cx, cy = self.params
        x, y = fx * local.x / depth + cx, cy - fy * local.y / depth
        return x, y, depth

    def ray(self, x, y):
        fx, fy, cx, cy = self.params
        local = Vector(((x - cx) / fx, -(y - cy) / fy, -1))
        length = local.length
        direction = (self.rotation @ local).normalized()
        # Clipping distances are camera-Z planes, not radial distances.
        near_distance = self.near * length
        origin = self.center + direction * near_distance
        return origin, direction, (self.far - self.near) * length

    def cast(self, tree, x, y):
        if tree is None:
            return None, None, None, None
        origin, direction, distance = self.ray(x, y)
        return tree.ray_cast(origin, direction, distance)


def render_objects(view_layer):
    """Objects reachable through at least one render-enabled collection path."""
    allowed = set()

    def visit(layer):
        if layer.exclude or layer.collection.hide_render:
            return
        allowed.update(obj.original for obj in layer.collection.objects
                       if not obj.hide_render)
        for child in layer.children:
            visit(child)

    visit(view_layer.layer_collection)
    return allowed


def _instance_collection_objects(collection, visited=None):
    """Honor ancestor render flags inside an instanced collection hierarchy."""
    visited = set() if visited is None else visited
    if collection in visited or collection.hide_render:
        return set()
    visited.add(collection)
    allowed = set()
    for obj in collection.objects:
        if not obj.hide_render:
            allowed.add(obj.original)
            if obj.instance_collection:
                allowed.update(_instance_collection_objects(obj.instance_collection, visited))
    for child in collection.children:
        allowed.update(_instance_collection_objects(child, visited))
    return allowed


@contextmanager
def geometry_visibility(context):
    """Evaluate render-visible objects even when hidden only in the viewport.

    Restore every flag before returning control to Blender's event loop. Render
    modifier differences are reported separately rather than silently changed.
    """
    changes = []
    hidden = []
    try:
        def reveal(value, attribute):
            if getattr(value, attribute):
                changes.append((value, attribute, True))
                setattr(value, attribute, False)

        def visit(layer):
            if layer.exclude or layer.collection.hide_render:
                return
            reveal(layer, "hide_viewport")
            reveal(layer.collection, "hide_viewport")
            for child in layer.children:
                visit(child)

        visit(context.view_layer.layer_collection)
        # Instanced collections can live outside the scene's collection tree.
        visited = set()

        def reveal_instance_collection(collection):
            if collection in visited or collection.hide_render:
                return
            visited.add(collection)
            reveal(collection, "hide_viewport")
            for obj in collection.objects:
                if not obj.hide_render:
                    reveal(obj, "hide_viewport")
                    if obj.instance_collection:
                        reveal_instance_collection(obj.instance_collection)
            for child in collection.children:
                reveal_instance_collection(child)

        for obj in render_objects(context.view_layer):
            reveal(obj, "hide_viewport")
            if obj.hide_get(view_layer=context.view_layer):
                hidden.append(obj)
                obj.hide_set(False, view_layer=context.view_layer)
            if obj.instance_collection:
                reveal_instance_collection(obj.instance_collection)
        context.view_layer.update()
        yield context.evaluated_depsgraph_get()
    finally:
        for obj in hidden:
            obj.hide_set(True, view_layer=context.view_layer)
        for value, attribute, original in reversed(changes):
            setattr(value, attribute, original)
        context.view_layer.update()


def _material_is_transparent(material):
    """Return whether a material can transmit or skip camera rays."""
    if material is None or not material.use_nodes or material.node_tree is None:
        return False
    for node in material.node_tree.nodes:
        if node.type in {
            "BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION", "BSDF_TRANSLUCENT",
        }:
            return True
        if node.type != "BSDF_PRINCIPLED":
            continue
        alpha = node.inputs.get("Alpha")
        if alpha is not None and (alpha.is_linked or float(alpha.default_value) < 0.999999):
            return True
        transmission = node.inputs.get("Transmission Weight") or node.inputs.get("Transmission")
        if transmission is not None and (transmission.is_linked or float(transmission.default_value) > 1e-6):
            return True
    return False


def build_geometry(context, skip_transparent=True):
    """Build a world-space BVH, optionally passing rays through glass faces."""
    vertices, triangles = [], []
    warnings = set()
    skipped_transparent_triangles = 0
    allowed = render_objects(context.view_layer)
    instance_sources = {obj: _instance_collection_objects(obj.instance_collection)
                        for obj in allowed if obj.instance_collection}
    checked = set()
    with geometry_visibility(context) as depsgraph:
        instances = []
        for instance in depsgraph.object_instances:
            obj = instance.object
            original = obj.original
            owner = instance.parent.original if instance.is_instance else original
            if owner not in allowed or original.hide_render or not instance.show_self:
                continue
            if not getattr(original, "visible_camera", True):
                continue
            if obj.type != "MESH":
                if obj.type in {"CURVE", "SURFACE", "FONT", "META", "VOLUME", "CURVES"}:
                    warnings.add(f"'{original.name}': non-mesh geometry is not ray sampled")
                continue
            if instance.is_instance and owner in instance_sources and original not in instance_sources[owner]:
                continue
            matrix = instance.matrix_world.copy()
            key = (owner.name_full, original.name_full, tuple(instance.persistent_id),
                   tuple(value for row in matrix for value in row))
            instances.append((key, obj, matrix))
        for _, obj, matrix in sorted(instances, key=lambda entry: entry[0]):
            original = obj.original
            if original not in checked:
                checked.add(original)
                for modifier in original.modifiers:
                    if modifier.show_viewport != modifier.show_render:
                        warnings.add(f"'{original.name}': modifier '{modifier.name}' differs between viewport and render")
                    if modifier.type == "SUBSURF" and modifier.levels != modifier.render_levels:
                        warnings.add(f"'{original.name}': subdivision viewport/render levels differ")
                for material in obj.data.materials:
                    if material and _material_is_transparent(material):
                        if skip_transparent:
                            warnings.add(f"'{material.name}': transparent/glass faces skipped so rays reach surfaces behind them")
                        else:
                            warnings.add(f"'{material.name}': transparent/glass faces included as opaque ray geometry")
            mesh = obj.to_mesh()
            try:
                if mesh is None:
                    continue
                mesh.calc_loop_triangles()
                offset = len(vertices)
                vertices.extend(matrix @ vertex.co for vertex in mesh.vertices)
                transparent_materials = {
                    material_index: _material_is_transparent(material)
                    for material_index, material in enumerate(mesh.materials)
                }
                for triangle in mesh.loop_triangles:
                    if skip_transparent and transparent_materials.get(triangle.material_index, False):
                        skipped_transparent_triangles += 1
                        continue
                    triangles.append(tuple(offset + index for index in triangle.vertices))
            finally:
                obj.to_mesh_clear()
        tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True) if triangles else None
    if context.scene.render.use_simplify:
        warnings.add("Render simplification is enabled; evaluated geometry may differ from rendering")
    if skipped_transparent_triangles:
        warnings.add(f"Skipped {skipped_transparent_triangles:,} transparent/glass triangle(s) from ray occlusion")
    return tree, sorted(warnings), len(triangles)


def collect_transparent_border_edges(context):
    """Return world-space edges on the boundary of transparent mesh faces.

    Transparent faces stay out of the ray BVH when pass-through mode is active,
    but their outer edges are still useful geometry for a reconstructed point
    cloud.  An edge is a border when exactly one transparent loop triangle uses
    it.  This removes triangulation diagonals while retaining pane outlines and
    transparent/opaque material boundaries.
    """
    edges = []
    allowed = render_objects(context.view_layer)
    instance_sources = {obj: _instance_collection_objects(obj.instance_collection)
                        for obj in allowed if obj.instance_collection}
    with geometry_visibility(context) as depsgraph:
        instances = []
        for instance in depsgraph.object_instances:
            obj = instance.object
            original = obj.original
            owner = instance.parent.original if instance.is_instance else original
            if owner not in allowed or original.hide_render or not instance.show_self:
                continue
            if not getattr(original, "visible_camera", True) or obj.type != "MESH":
                continue
            if instance.is_instance and owner in instance_sources and original not in instance_sources[owner]:
                continue
            matrix = instance.matrix_world.copy()
            key = (owner.name_full, original.name_full, tuple(instance.persistent_id),
                   tuple(value for row in matrix for value in row))
            instances.append((key, obj, matrix))

        for _, obj, matrix in sorted(instances, key=lambda entry: entry[0]):
            mesh = obj.to_mesh()
            try:
                if mesh is None:
                    continue
                mesh.calc_loop_triangles()
                transparent_materials = {
                    material_index: _material_is_transparent(material)
                    for material_index, material in enumerate(mesh.materials)
                }
                edge_counts = {}
                for triangle in mesh.loop_triangles:
                    if not transparent_materials.get(triangle.material_index, False):
                        continue
                    indices = triangle.vertices
                    for first, second in ((indices[0], indices[1]),
                                          (indices[1], indices[2]),
                                          (indices[2], indices[0])):
                        edge = (min(first, second), max(first, second))
                        edge_counts[edge] = edge_counts.get(edge, 0) + 1
                for (first, second), count in edge_counts.items():
                    if count != 1:
                        continue
                    edges.append((
                        tuple(matrix @ mesh.vertices[first].co),
                        tuple(matrix @ mesh.vertices[second].co),
                    ))
            finally:
                obj.to_mesh_clear()
    return edges


class ImagePixels:
    """Read saved RGB values without applying another color/view transform."""

    def __init__(self, path, width, height):
        self.image = None
        self.pixels = None
        self.width, self.height = width, height
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required image is missing or empty: {path}")
        try:
            self.image = bpy.data.images.load(str(path), check_existing=False)
            # A private image datablock avoids changing an image used by the scene.
            self.image.colorspace_settings.name = "Non-Color"
            self.image.alpha_mode = "STRAIGHT"
            if tuple(self.image.size) != (width, height):
                raise ValueError(f"Image '{path.name}' is {tuple(self.image.size)}, expected {width} x {height}")
            self.channels = self.image.channels
            if self.channels not in {1, 3, 4} or len(self.image.pixels) != width * height * self.channels:
                raise ValueError(f"Image cannot be decoded as RGB: {path}")
            self.pixels = array('f', [0]) * len(self.image.pixels)
            self.image.pixels.foreach_get(self.pixels)
        except Exception:
            self.close()
            raise

    def rgb(self, x, y):
        column = min(self.width - 1, max(0, math.floor(x)))
        row = self.height - 1 - min(self.height - 1, max(0, math.floor(y)))
        offset = (row * self.width + column) * self.channels
        return tuple(max(0, min(255, round(self.pixels[offset + (c if self.channels > 1 else 0)] * 255)))
                     for c in range(3))

    def close(self):
        self.pixels = None
        if self.image is not None:
            bpy.data.images.remove(self.image)
            self.image = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Observations:
    """Compact per-image observations, preserving zero-based COLMAP indices."""

    def __init__(self):
        self.xy = array('d')
        self.ids = array('Q')

    def __len__(self):
        return len(self.ids)

    def append(self, x, y, point_id):
        self.xy.extend((x, y))
        self.ids.append(point_id)

    def __iter__(self):
        for index, point_id in enumerate(self.ids):
            yield self.xy[2 * index], self.xy[2 * index + 1], point_id


class PointBuffer:
    """Packed point/track arrays plus collision-safe spatial hash buckets.

    Memory grows with retained points and observations, never with all possible
    point-camera pairs. Python dicts are used only for spatial bucket heads.
    """

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.positions = array('d')
        self.colors = array('B')
        self.quality = array('d')
        self.triangles = array('q')
        self.bucket_next = array('q')
        self.track_heads = array('q')
        self.last_image = array('q')
        self.track_image = array('Q')
        self.track_index = array('Q')
        self.track_next = array('q')
        self.buckets = {}

    def __len__(self):
        return len(self.triangles)

    def position(self, index):
        offset = 3 * index
        return tuple(self.positions[offset:offset + 3])

    def _cell(self, position):
        return tuple(math.floor(value / self.tolerance) for value in position)

    def find(self, position, triangle):
        x, y, z = self._cell(position)
        limit = self.tolerance ** 2
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    index = self.buckets.get(hash((triangle, x + dx, y + dy, z + dz)), -1)
                    while index != -1:
                        if self.triangles[index] == triangle:
                            offset = 3 * index
                            distance = sum((self.positions[offset + c] - position[c]) ** 2 for c in range(3))
                            if distance <= limit and (best is None or index < best):
                                best = index
                        index = self.bucket_next[index]
        return best

    def add(self, position, triangle):
        index = len(self)
        key = hash((triangle, *self._cell(position)))
        self.positions.extend(position)
        self.colors.extend((0, 0, 0))
        self.quality.append(float('inf'))
        self.triangles.append(triangle)
        self.bucket_next.append(self.buckets.get(key, -1))
        self.buckets[key] = index
        self.track_heads.append(-1)
        self.last_image.append(-1)
        return index

    def observe(self, point, image_id, point2d_index, color, footprint):
        if self.last_image[point] == image_id:
            return False
        self.last_image[point] = image_id
        track = len(self.track_image)
        self.track_image.append(image_id)
        self.track_index.append(point2d_index)
        self.track_next.append(self.track_heads[point])
        self.track_heads[point] = track
        if footprint < self.quality[point]:
            self.quality[point] = footprint
            self.colors[3 * point:3 * point + 3] = array('B', color)
        return True

    def tracks(self, point):
        track = self.track_heads[point]
        while track != -1:
            yield self.track_image[track], self.track_index[track]
            track = self.track_next[track]

    def __iter__(self):
        for index in range(len(self)):
            yield {"position": self.position(index),
                   "color": tuple(self.colors[3 * index:3 * index + 3]),
                   "tracks": tuple(self.tracks(index))}

    def release_spatial_index(self):
        self.buckets.clear()
        self.bucket_next = array('q')


def sample_camera(tree, frame, pixels, points, observations, image_id, spacing, tile_size=2048):
    """Yield completed-ray counts between tiles so the UI can cancel safely."""
    completed = 0
    for y in range(0, frame.height, spacing):
        for x in range(0, frame.width, spacing):
            location, normal, triangle, _ = frame.cast(tree, x + 0.5, y + 0.5)
            completed += 1
            if location is not None:
                candidate = points.find(location, triangle)
                pixel = None
                if candidate is not None:
                    retained = points.position(candidate)
                    pixel = frame.project(retained)
                    if pixel is not None:
                        hit, _, hit_triangle, _ = frame.cast(tree, pixel[0], pixel[1])
                        # A distance match alone would accept a nearby occluder.
                        epsilon = max(1e-6, (Vector(retained) - frame.center).length * 2e-6)
                        if hit is None or hit_triangle != triangle or (hit - Vector(retained)).length > epsilon:
                            pixel = None
                    if pixel is None:
                        candidate = None
                if candidate is None:
                    pixel = frame.project(location)
                    if pixel is not None:
                        candidate = points.add(location, triangle)
                if candidate is not None and pixel is not None and points.last_image[candidate] != image_id:
                    direction = (Vector(points.position(candidate)) - frame.center).normalized()
                    fx, fy, _, _ = frame.params
                    footprint = pixel[2] ** 2 / (fx * fy * max(abs(normal.dot(direction)), 1e-8))
                    if points.observe(candidate, image_id, len(observations), pixels.rgb(*pixel[:2]), footprint):
                        observations.append(pixel[0], pixel[1], candidate + 1)
            if completed % tile_size == 0:
                yield completed
    yield completed


MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def commit_model(stage, output):
    """Replace all model files, rolling back handled failures during replacement."""
    backups, installed = [], []
    try:
        for name in MODEL_FILES:
            target = output / name
            if target.exists() and not target.is_file():
                raise ValueError(f"Model output is not a regular file: {target}")
            if target.exists():
                backup = stage / (name + '.previous')
                target.replace(backup)
                backups.append((backup, target))
            (stage / name).replace(target)
            installed.append(target)
    except Exception:
        for target in installed:
            target.unlink()
        for backup, target in reversed(backups):
            backup.replace(target)
        raise
