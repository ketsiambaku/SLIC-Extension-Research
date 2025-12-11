# Dynamic SLIC Superpixels Research
Research project that tests some potential ways to improve SLIC superpixel algorithm and ways to use SLIC superpixels to do reverse image searching. University of Washington Bothell CSS 587 (Advanced Topics in Computer Vision) Autumn 2025 quarter research project.

Collaborators: 
Chandler Calkins
Ketsia Mbaku
Everett-Alan Hood
Luke Erdy

# Final Deliverables
The ``main.cpp`` file in the ``SuperpixelImageSearch/src`` folder is the primary final deliverable of the project. It combines all parts of the project and shows how SLIC superpixels can be used to do reverse image searching. Some of the other folders also contain demonstrations of the other parts of the project. The sections below describe each demonstration and how to use it.

## SuperpixelImageSearch
By: Everett-Alan Hood

This module implements the reverse image search system used in our project, including both the standalone SuperpixelImageSearch algorithm and the full integrated SD-SLIC pipeline demo.

If you would like to see results on a larger dataset (10,000 images) without running the full experiment yourself, precomputed outputs are available in: ``SuperpixelImageSearch/output/``

1. Downloading the Sample COCO Dataset (2500 Images)
    Before running the programs, download the small COCO subset used for testing.
    From the project root: 
    ``cd SuperpixelImageSearch/scripts``
    ``python download_coco.py``
2. Running the Pipeline Demo
    To see the full SD-SLIC pipeline (superpixels → merged regions → descriptors → indexing → retrieval), run:
    ``cd build/Debug``
    ``./pipeline_demo.exe``
    This demo produces:
    - SD-SLIC superpixel boundaries
    - Region-level SIFT descriptors
    - Top-K nearest matches
    - Pipeline visualizations and CSV logs
    
    Outputs are written to: ``SuperpixelImageSearch/output/pipeline_demo/``

3. Running the Main Reverse Image Search Program
    To benchmark the actual SuperpixelImageSearch algorithm and compare descriptor types, run:
    ``cd build/Debug``
    ``./superpixel_ris.exe``
    This program evaluates:
    - Global SIFT / ORB descriptors
    - Superpixel-spatial descriptors (multiple grid sizes)
    - Custom SD-SLIC region descriptors

    Results appear under: ```SuperpixelImageSearch/output/```

## LTRIDP x SDP
By: Ketsia Mbaku

### Prerequisites
- The project must be built from the root using CMake so that all dependencies are available.

### Build Instructions
**Important:** You must build the `ltridp` libraries first, as `SDP_LTRIDP` depends on them.

From the project root, run:

```sh
# 1. Build ltridp libraries (required)
cmake -S ltridp -B ltridp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ltridp/build

# 2. Build SDP_LTRIDP test target
cmake -S SDP_LTRIDP -B SDP_LTRIDP/build -DCMAKE_BUILD_TYPE=Release
cmake --build SDP_LTRIDP/build --target test_complete_pipeline_v2
```

### Running the Test

To see the full LTriDP pipeline combined with SuperDuperPixel in action, run:

```sh
./SDP_LTRIDP/build/tests/test_complete_pipeline_v2 ltridp/data/input output_sdp_ltridp
```

- `ltridp/data/input`: Directory containing input images (PNG, JPG, etc.)
- `output_sdp_ltridp`: Directory where output images and results will be saved. You can specify any folder. If it does not exists, the tests will create it

### Output
For each input image and region size, the test will generate:
- Superpixel boundary images
- Duperized superpixel boundary images
- Pipeline comparison grids
- Console output edge scores and compactness metrics

### Notes
- the input directory exists under `ltridp/data/input` and contains 10 .png images before running the test. You can add more images from WBA if you wish.
- The output directory will be created if it does not exist.

## SDP_HashTable
By: Luke Erdy

``SDP_HashTable/src/HashTableDemo.cpp`` uses the class defined in ``SLICHashTable.hpp``  to segment images according to the modified SLIC algorithm (SDP), generate key-value pairs storing their superpixels in a hash map, and perform a sequence of queries which will return similar images from the present database based on similarity.
Superpixel hash keys are generated based on average color value and spatial extent. Best matches are found via a voting system which counts collisions on a simulated hash of a query image's superpixels.
To run this demo:
1. Place the header and source files found in ``SDP_HashTable/src`` in your Visual Studio Code project ``src`` folder.
2. Place the images found in ``SDP_HashTable/src/images`` in your main project folder (2 directories above ``build/Release``.)
3. Compile and run the program.
4. Optionally, you may add your own inputs and queries according to the naming scheme. The variables ``int input_count`` and ``int q_count`` must be changed to reflect the correct number of inputs and queries, respectively.

## SuperDuperPixels
By: Chandler Calkins

The ``src/demo.cpp`` file in this folder demonstrates generating super-duper-pixels from superpixels generated by SLIC.
It works by combining adjacent superpixels into one superpixel if they have either similar enough average colors of the pixels contained in them, or similar enough color histograms of the pixels in them.
It determines if adjacent superpixels have similar enough colors by measuring the manhattan distance (for speed, rather than euclidian distance) between the average colors or the color histograms.
If the distance is below a threshold, they are combined.
Color histograms have to be normalized between 0 and 1 for this to work with them so percentages of pixel colors are measured rather than total sums of pixel colors.
If this is not done, superpixels that are similar in color but different in size would have large distances between them and not be considered similar.

To run this demo, follow these steps:
1. Place an image in the folder that's either of the format .jpg, .png, or .gif and uses RGB color space.
2. Rename that image to either ``inputs.jpg``, ``input.png``, or ``inpug.gif`` depending on the file format.
3. If you are running this on Visual Studio instead of another IDE, comment out line `72` in ``src/demo.cpp`` so the program can file your input image. The line of code you need to comment out is: ``chdir("../../");``.
4. Run the program.
5. Observe the regular superpixels generated for the image by SLIC.
6. Press space to move to the next image.
7. Observe the super-duper-pixels generated for the image using the average colors of superpixels with the parameters used in the demo.
8. Press space to  move to the next image.
9. Observe the super-duper-pixels generated for the image using color histograms of superpixels with the parameters used in the demo.
10. Press space again to finish.
11. Optionally, observe the outputted files of each of the images that were shown named ``superpixels.png``, ``superduperpixels_average.png``, and ``superduperpixels_histogram.png``.
