# Release

1. Update git hash in CMakeLists.txt to point to the correct version of `qoco`
2. Update version number in `pyproject.toml` and `backend/cuda/pyproject.toml`
3. Run `./build_cuda_wheels.sh` to build `qoco-cuda` wheels
4. Download `qoco` wheels from `Build Wheels` workflow in `Actions` tab
5. Upload both wheels to `pypi`
