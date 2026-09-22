# Contributing

Thanks for helping improve Gaussian Camera Rig.

## Development setup

The add-on itself has no third-party Python dependencies. Install Blender
locally to run the integration suite; the supported minimum is Blender 3.6.

From the repository root, build the installable archive with:

```powershell
python scripts/build_addon_zip.py
```

This writes `gaussian_camera_rig.zip` with deterministic contents and excludes
Python bytecode and local cache files.

## Validation

Run the fast source and package checks with:

```powershell
python -m compileall -q gaussian_camera_rig tests
python scripts/build_addon_zip.py --output $env:TEMP\gaussian_camera_rig.zip
python scripts/validate_package.py $env:TEMP\gaussian_camera_rig.zip
```

Run the Blender integration suite separately:

```powershell
$env:GS_TEST_OUTPUT = Join-Path $env:TEMP 'gs-colmap-fixtures'
blender --background --factory-startup --python-exit-code 1 --python tests/blender_validation.py
```

If `pycolmap` is available in a test-only Python environment, validate the
generated COLMAP model as well:

```powershell
python tests/validate_colmap.py $env:GS_TEST_OUTPUT
```

## Pull requests

- Keep changes focused and explain user-visible behavior in the pull request.
- Update the README when installation, controls, output files, or supported
  versions change.
- Add or update Blender integration coverage for exporter or camera behavior.
- Do not commit `.blend` files, generated fixtures, caches, or personal editor
  settings.
