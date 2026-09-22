"""Run with blender --background --factory-startup --python-exit-code 1 --python this_file.

GS_TEST_OUTPUT can name a persistent fixture directory for an external COLMAP
reader. These tests create their own scenes and never open or save user .blend files.
"""
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
import zlib

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaussian_camera_rig as addon
from gaussian_camera_rig.raycast_export import (
    CameraFrame, ImagePixels, Observations, PointBuffer, build_geometry,
    commit_model, sample_camera, MODEL_FILES,
)

OUTPUT = Path(os.environ.get('GS_TEST_OUTPUT', tempfile.mkdtemp(prefix='gs-colmap-tests-')))
OUTPUT.mkdir(parents=True, exist_ok=True)


def png(path, width, height, color=lambda x, y: (x % 256, y % 256, 128)):
    """Write known encoded RGB bytes independently of Blender color management."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    data = b''.join(b'\0' + bytes(c for x in range(width) for c in color(x, y)) for y in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(data)) + chunk(b'IEND', b''))


def peak_memory_mb():
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ('peak', 'working', 'paged_peak', 'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.peak / 2**20


class RayExportTests(unittest.TestCase):
    def setUp(self):
        self.scene = bpy.context.scene
        for obj in tuple(self.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.scene.render.resolution_x = 32
        self.scene.render.resolution_y = 24
        self.scene.render.resolution_percentage = 100
        self.scene.render.pixel_aspect_x = self.scene.render.pixel_aspect_y = 1
        self.scene.render.use_border = False
        self.scene.render.use_multiview = False
        self.scene.render.use_simplify = False
        self.scene.camera = None
        self.camera = self.add_camera('Camera')
        self.scene.camera = self.camera
        self.directory = OUTPUT / self._testMethodName
        self.directory.mkdir(exist_ok=True)
        settings = self.scene.gs_colmap_export
        settings.output_path = str(self.directory)
        settings.render_images = False
        settings.skip_existing_images = True
        settings.render_format = 'PNG'
        settings.export_mesh_points = True
        settings.point_source = 'RAYS'
        settings.pixel_spacing = 1
        settings.merge_distance = 1e-6

    def add_camera(self, name, position=(0, 0, 0)):
        camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        self.scene.collection.objects.link(camera)
        camera.location = position
        camera.data.clip_start = .1
        camera.data.clip_end = 100
        return camera

    def plane(self, z=-5, size=20, name='Plane'):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(-size, -size, z), (size, -size, z), (size, size, z), (-size, size, z)], [], [(0, 1, 2, 3)])
        obj = bpy.data.objects.new(name, mesh)
        self.scene.collection.objects.link(obj)
        return obj

    def frame(self, camera=None):
        bpy.context.view_layer.update()
        return CameraFrame(camera or self.camera, bpy.context.evaluated_depsgraph_get(), self.scene,
                           *addon._render_dimensions(self.scene))

    def test_view_relative_camera_grid(self):
        settings = self.scene.gs_camera_rig
        settings.preset = 'GRID_2D'
        settings.camera_count = 12
        settings.grid_x_range = math.radians(180)
        settings.grid_y_range = math.radians(120)

        offsets = addon._grid_offsets(settings)
        directions = addon._directions(settings)
        self.assertEqual(len(offsets), 12)
        self.assertEqual(len(directions), 12)
        self.assertAlmostEqual(offsets[0][0], -1.0, places=6)
        self.assertAlmostEqual(offsets[0][1], -2.0 / 3.0, places=6)
        self.assertAlmostEqual(offsets[-1][0], 1.0, places=6)
        self.assertAlmostEqual(offsets[-1][1], 2.0 / 3.0, places=6)
        self.assertTrue(all(direction == Vector((0.0, 1.0, 0.0)) for direction in directions))

    def images(self, color=lambda x, y: (x, y, 128)):
        width, height = addon._render_dimensions(self.scene)
        for camera in addon._scene_camera_objects(self.scene):
            png(self.directory / 'images' / addon._safe_image_name(camera, 'png'), width, height, color)

    def export(self):
        return bpy.ops.gs_camera_rig.export_colmap('EXEC_DEFAULT')

    def read_model(self):
        images, points = {}, {}
        lines = iter((self.directory / 'images.txt').read_text().splitlines())
        for line in lines:
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            obs = next(lines).split()
            images[int(fields[0])] = [(float(obs[i]), float(obs[i + 1]), int(obs[i + 2]))
                                      for i in range(0, len(obs), 3)]
        for line in (self.directory / 'points3D.txt').read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            points[int(fields[0])] = (tuple(map(float, fields[1:4])), tuple(map(int, fields[4:7])),
                                      list(zip(map(int, fields[8::2]), map(int, fields[9::2]))))
        for point_id, (_, _, tracks) in points.items():
            self.assertTrue(tracks)
            self.assertEqual(len(tracks), len({image for image, _ in tracks}))
            for image_id, index in tracks:
                self.assertEqual(images[image_id][index][2], point_id)
        for image_id, observations in images.items():
            for index, (_, _, point_id) in enumerate(observations):
                self.assertIn((image_id, index), points[point_id][2])
        return images, points

    def test_plane_and_repeated_cameras(self):
        self.plane()
        self.add_camera('Camera2')
        self.images()
        self.assertEqual(self.export(), {'FINISHED'})
        images, points = self.read_model()
        self.assertEqual(len(images), 2)
        self.assertEqual(len(points), 32 * 24)
        self.assertTrue(all(len(point[2]) == 2 for point in points.values()))
        for point_id, (position, color, _) in points.items():
            self.assertAlmostEqual(position[2], -5, places=5)
        for x, y, point_id in images[1]:
            self.assertEqual(points[point_id][1], (math.floor(x), math.floor(y), 128))
        before = {name: (self.directory / name).read_bytes() for name in MODEL_FILES}
        self.export()
        self.assertEqual(before, {name: (self.directory / name).read_bytes() for name in before})

    def test_occlusion_and_thin_surfaces(self):
        self.plane()
        self.plane(z=-3, size=.5, name='Foreground')
        self.plane(z=-3.0001, size=.5, name='BehindThinSurface')
        self.images()
        self.export()
        _, points = self.read_model()
        depths = [position[2] for position, _, _ in points.values()]
        self.assertTrue(any(abs(z + 3) < 1e-6 for z in depths))
        self.assertTrue(any(abs(z + 5) < 1e-6 for z in depths))
        self.assertFalse(any(abs(z + 3.0001) < 1e-6 for z in depths))

    def test_glass_is_skipped_and_can_be_included(self):
        glass = self.plane(z=-3, size=20, name='Glass')
        self.plane(z=-5, size=20, name='Background')
        material = bpy.data.materials.new('Glass material')
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in tuple(nodes):
            nodes.remove(node)
        output = nodes.new('ShaderNodeOutputMaterial')
        glass_node = nodes.new('ShaderNodeBsdfGlass')
        glass_node.inputs['Roughness'].default_value = 0.0
        material.node_tree.links.new(glass_node.outputs['BSDF'], output.inputs['Surface'])
        glass.data.materials.append(material)
        tree, warnings, count = build_geometry(bpy.context, skip_transparent=True)
        hit = self.frame().cast(tree, 16.5, 12.5)[0]
        self.assertAlmostEqual(hit.z, -5, places=5)
        self.assertTrue(any('Glass material' in warning for warning in warnings))
        self.assertEqual(count, 2)
        tree, _, count = build_geometry(bpy.context, skip_transparent=False)
        hit = self.frame().cast(tree, 16.5, 12.5)[0]
        self.assertAlmostEqual(hit.z, -3, places=5)
        self.assertEqual(count, 4)

    def test_glass_border_points_are_exported(self):
        glass = self.plane(z=-3, size=.8, name='Glass')
        self.plane(z=-5, size=20, name='Background')
        material = bpy.data.materials.new('Glass border material')
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in tuple(nodes):
            nodes.remove(node)
        output = nodes.new('ShaderNodeOutputMaterial')
        glass_node = nodes.new('ShaderNodeBsdfGlass')
        glass_node.inputs['Roughness'].default_value = 0.0
        material.node_tree.links.new(glass_node.outputs['BSDF'], output.inputs['Surface'])
        glass.data.materials.append(material)
        self.images()

        self.export()
        _, points = self.read_model()
        border_points = [point for point in points.values() if abs(point[0][2] + 3) < 1e-6]
        self.assertGreaterEqual(len(border_points), 4)
        self.assertTrue(all(point[2] for point in border_points))

        self.scene.gs_colmap_export.include_glass_borders = False
        self.export()
        _, points = self.read_model()
        self.assertFalse(any(abs(point[0][2] + 3) < 1e-6 for point in points.values()))

    def test_projection(self):
        parent = bpy.data.objects.new('Parent', None)
        self.scene.collection.objects.link(parent)
        self.camera.parent = parent
        for width, height, aspect, fit in [(640, 480, (1, 1), 'AUTO'), (320, 640, (1, 2), 'AUTO'),
                                           (640, 320, (2, 1), 'VERTICAL'), (317, 213, (1, 1), 'HORIZONTAL')]:
            for shift_x, shift_y in [(0, 0), (.2, -.1)]:
                self.scene.render.resolution_x, self.scene.render.resolution_y = width, height
                self.scene.render.pixel_aspect_x, self.scene.render.pixel_aspect_y = aspect
                self.camera.data.sensor_fit = fit
                self.camera.data.shift_x, self.camera.data.shift_y = shift_x, shift_y
                parent.scale = (2, 3, 4)
                self.camera.scale = (1.5, .5, 2)
                parent.rotation_euler = (.2, -.3, .4)
                parent.location = (1, 2, 3)
                frame = self.frame()
                for x, y in [(0.5, .5), (width / 2, height / 2), (width - .5, height - .5)]:
                    origin, direction, _ = frame.ray(x, y)
                    position = origin + direction * 10
                    projection = frame.project(position)
                    expected = world_to_camera_view(self.scene, self.camera, position)
                    self.assertAlmostEqual(projection[0], expected.x * width, delta=.002)
                    self.assertAlmostEqual(projection[1], (1 - expected.y) * height, delta=.002)
                    self.assertAlmostEqual(projection[0], x, delta=.002)
                    self.assertAlmostEqual(projection[1], y, delta=.002)
                old = addon._camera_intrinsics(self.camera.data, self.scene, width, height)
                for a, b in zip(old, frame.params):
                    self.assertAlmostEqual(a, b, delta=.002)

    def test_modifiers_instances_visibility(self):
        plane = self.plane()
        solidify = plane.modifiers.new('Thickness', 'SOLIDIFY')
        solidify.thickness = 1
        solidify.offset = 1
        plane.hide_set(True)
        plane.hide_viewport = True
        tree, _, _ = build_geometry(bpy.context)
        self.assertTrue(plane.hide_viewport)
        self.assertTrue(plane.hide_get())
        self.assertAlmostEqual(self.frame().cast(tree, 16.5, 12.5)[0].z, -4, places=5)
        plane.hide_render = True
        tree, _, count = build_geometry(bpy.context)
        self.assertEqual(count, 0)
        collection = bpy.data.collections.new('InstanceSource')
        source = self.plane(z=0, size=2)
        self.scene.collection.objects.unlink(source)
        collection.objects.link(source)
        instance = bpy.data.objects.new('Instancer', None)
        instance.instance_type = 'COLLECTION'
        instance.instance_collection = collection
        instance.location = (0, 0, -7)
        instance.scale = (2, 3, 1)
        self.scene.collection.objects.link(instance)
        tree, _, count = build_geometry(bpy.context)
        self.assertGreater(count, 0)
        self.assertAlmostEqual(self.frame().cast(tree, 16.5, 12.5)[0].z, -7, places=5)
        # A hidden ancestor inside an instance must exclude its descendants.
        child = bpy.data.collections.new('InstanceChild')
        collection.children.link(child)
        collection.objects.unlink(source)
        child.objects.link(source)
        collection.hide_render = True
        tree, _, count = build_geometry(bpy.context)
        self.assertEqual(count, 0)

    def test_clip_planes_and_empty_scene(self):
        self.plane(z=-.05)
        self.plane(z=-5)
        self.camera.data.clip_end = 4
        self.images()
        self.export()
        self.assertEqual(len(self.read_model()[1]), 0)
        self.camera.data.clip_end = 6
        self.export()
        self.assertEqual(len(self.read_model()[1]), 32 * 24)

    def test_missing_and_wrong_images_preserve_model(self):
        (self.directory / 'images' / 'Camera.png').unlink(missing_ok=True)
        for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
            (self.directory / name).write_text('previous model')
        before_images = set(bpy.data.images)
        with self.assertRaisesRegex(RuntimeError, 'Required image'):
            self.export()
        png(self.directory / 'images' / 'Camera.png', 2, 2)
        with self.assertRaisesRegex(RuntimeError, 'expected 32 x 24'):
            self.export()
        (self.directory / 'images' / 'Camera.png').write_bytes(b'not an image')
        with self.assertRaises(RuntimeError):
            self.export()
        self.assertEqual(set(bpy.data.images), before_images)
        self.assertIsNone(addon.active_export)
        for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
            self.assertEqual((self.directory / name).read_text(), 'previous model')

    def test_cancellation_cleanup(self):
        self.plane()
        self.images()
        original = (self.scene.camera, self.scene.render.filepath, self.scene.render.image_settings.file_format)
        for name in MODEL_FILES:
            (self.directory / name).write_text('previous')
        before = set(bpy.data.images)
        for phase in ('Rays', 'Writing model'):
            steps = addon.GS_OT_export_colmap._export_steps(Harness(), bpy.context)
            for _, message in steps:
                if message.startswith(phase):
                    break
            steps.close()
            self.assertEqual(before, set(bpy.data.images))
            self.assertEqual(original, (self.scene.camera, self.scene.render.filepath, self.scene.render.image_settings.file_format))
            self.assertFalse(list(self.directory.glob('.gs-colmap-*')))
            for name in MODEL_FILES:
                self.assertEqual((self.directory / name).read_text(), 'previous')

    def test_commit_rollback(self):
        stage = self.directory / 'stage'
        stage.mkdir(exist_ok=True)
        for name in MODEL_FILES:
            (self.directory / name).write_text('previous')
        (stage / 'cameras.txt').write_text('new cameras')
        # An absent staged images.txt simulates a replacement failure midway.
        with self.assertRaises(FileNotFoundError):
            commit_model(stage, self.directory)
        for name in MODEL_FILES:
            self.assertEqual((self.directory / name).read_text(), 'previous')

    def test_color_view_selection(self):
        self.plane()
        equal = self.add_camera('CameraB')
        closer = self.add_camera('CameraC', (0, 0, -1))
        closer.data.lens = 120
        self.scene.gs_colmap_export.merge_distance = .00001
        for camera, color in [(self.camera, (255, 0, 0)), (equal, (0, 0, 255)), (closer, (0, 255, 0))]:
            png(self.directory / 'images' / addon._safe_image_name(camera, 'png'), 32, 24, lambda x, y: color)
        self.export()
        _, points = self.read_model()
        shared = [point for point in points.values() if len(point[2]) == 3]
        self.assertTrue(shared)
        self.assertTrue(all(color == (0, 255, 0) for _, color, _ in shared))
        tied = [point for point in points.values() if {image for image, _ in point[2]} == {1, 2}]
        self.assertTrue(tied)
        self.assertTrue(all(color == (255, 0, 0) for _, color, _ in tied))

    def test_render_and_reuse(self):
        self.plane()
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = 1
        self.scene.cycles.device = 'CPU'
        self.scene.render.resolution_x, self.scene.render.resolution_y = 65, 49
        self.scene.render.resolution_percentage = 50
        settings = self.scene.gs_colmap_export
        settings.render_images = True
        # A bad existing image must be replaced, never reused.
        png(self.directory / 'images' / 'Camera.png', 2, 2)
        original = (self.scene.camera, self.scene.render.filepath, self.scene.render.image_settings.file_format,
                    self.scene.render.use_file_extension)
        self.export()
        self.assertEqual(original, (self.scene.camera, self.scene.render.filepath, self.scene.render.image_settings.file_format,
                                   self.scene.render.use_file_extension))
        self.assertEqual(len(self.read_model()[1]), 32 * 24)
        image = self.directory / 'images' / 'Camera.png'
        mtime = image.stat().st_mtime_ns
        self.export()
        self.assertEqual(mtime, image.stat().st_mtime_ns)
        settings.render_format = 'JPEG'
        self.export()
        with ImagePixels(self.directory / 'images' / 'Camera.jpg', 32, 24) as pixels:
            self.assertEqual(len(pixels.rgb(1.5, 1.5)), 3)

    def test_spacing_and_no_geometry(self):
        self.images()
        self.export()
        self.assertEqual(len(self.read_model()[1]), 0)
        self.plane()
        self.scene.gs_colmap_export.pixel_spacing = 3
        self.export()
        self.assertEqual(len(self.read_model()[1]), math.ceil(32 / 3) * math.ceil(24 / 3))

    def test_legacy_and_disabled_points(self):
        self.plane(size=1)
        self.scene.gs_colmap_export.point_source = 'VERTICES'
        self.export()
        self.assertEqual(len(self.read_model()[1]), 4)
        self.scene.gs_colmap_export.export_mesh_points = False
        self.export()
        self.assertEqual(len(self.read_model()[1]), 0)

    def test_spatial_hash_boundaries(self):
        points = PointBuffer(.1)
        one = points.add((.099, 0, 0), 1)
        self.assertEqual(points.find((.101, 0, 0), 1), one)
        self.assertIsNone(points.find((.101, 0, 0), 2))
        self.assertIsNone(points.find((.21, 0, 0), 1))

    def test_blender_52_timer_event_compatibility(self):
        class EventWithoutTimer:
            type = 'TIMER'

        class EventWithTimer:
            type = 'TIMER'
            timer = object()

        timer = object()
        self.assertTrue(addon._is_export_timer_event(EventWithoutTimer(), timer))
        self.assertTrue(addon._is_export_timer_event(EventWithTimer(), EventWithTimer.timer))
        self.assertFalse(addon._is_export_timer_event(EventWithTimer(), timer))
        self.assertFalse(addon._is_export_timer_event(type('Event', (), {'type': 'MOUSEMOVE'})(), timer))

    def test_border_and_multiview_are_temporarily_disabled(self):
        self.plane()
        self.images()
        self.scene.render.use_border = True
        self.scene.render.border_min_x = .2
        self.scene.render.border_max_x = .8
        self.scene.render.border_min_y = .1
        self.scene.render.border_max_y = .9
        self.scene.render.use_crop_to_border = True
        self.scene.render.use_multiview = True
        self.export()
        self.assertTrue(self.scene.render.use_border)
        self.assertTrue(self.scene.render.use_crop_to_border)
        self.assertTrue(self.scene.render.use_multiview)
        self.assertEqual(len(self.read_model()[1]), 32 * 24)

    def test_z_dense_benchmark(self):
        self.scene.render.resolution_x, self.scene.render.resolution_y = 512, 384
        self.plane()
        self.add_camera('Camera2', (.25, 0, 0))
        self.images(lambda x, y: (x % 256, y % 256, 128))
        started = time.perf_counter()
        self.export()
        elapsed = time.perf_counter() - started
        peak = peak_memory_mb()
        images, points = self.read_model()
        result = {'rays': 512 * 384 * 2, 'points': len(points), 'seconds': elapsed,
                  'peak_process_memory_mb': peak, 'blender': bpy.app.version_string}
        (OUTPUT / 'benchmark.json').write_text(json.dumps(result, indent=2))
        print('BENCHMARK', json.dumps(result))
        self.assertEqual(sum(map(len, images.values())), 512 * 384 * 2)


class Harness:
    def report(self, level, message):
        print(level, message)


addon.register()
print('TEST_OUTPUT', OUTPUT)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(RayExportTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
addon.unregister()
if not result.wasSuccessful():
    raise RuntimeError('Blender validation failed')
