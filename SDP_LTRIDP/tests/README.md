# SDP-LTriDP Test Utilities

This directory contains the end-to-end regression drivers for the improved SLIC pipeline. The primary entry point is `test_complete_pipeline_v2`, which reproduces all qualitative figures referenced in the report.

## Prerequisites
- CMake 3.20+ and a C++20-capable compiler (tested with AppleClang 16 and GCC 13)
- OpenCV 4.12 with the `ximgproc` module available in your toolchain
- MRI slices stored as 8-bit grayscale PNG/JPG/BMP images

## Build Instructions
From the repository root run:

```bash
cmake -S . -B build
cmake --build build --target test_complete_pipeline_v2
```

The compiled binary will be placed under `build/SDP_LTRIDP/tests/`.

## Running the Complete Pipeline
Invoke the executable with an input directory containing MRI images and an output directory for artifacts:

```bash
./build/SDP_LTRIDP/tests/test_complete_pipeline_v2 \
    ./ltridp/data/input \
    ./ltridp/data/output_v2
```

Each image is processed with region sizes `S in {5, 10, 20, 30}`. For every setting the program saves three files per image - `*_boundaries.png`, `*_duperized.png`, and `*_pipeline.png` - inside the output directory. Create the output directory ahead of time if you want to keep previous runs separate.

Use `--help` or run the executable without arguments to see the usage banner printed by the program.
