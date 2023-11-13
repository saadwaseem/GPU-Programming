# Celestial Object Analysis

## Problem Description

The project was about performing an analysis on celestial object data, specifically calculating angular correlation functions for real and simulated galaxies. The input consists of right ascension (RA) and declination (DEC) values for both real and simulated galaxies (100K in each case). The analysis involves comparing the distributions of these galaxies to understand their clustering behavior and learn GPU programming by solving a scientific problem.

## Parallelism Implementation

The calculation of DD, RR, DR is a O(n^2) operation.
For a catalog of n galaxies, there are n(n − 1)/2 calculations. Same formula is applied over all n elements for each set of angles.Parallelism is implemented through CUDA kernels, specifically the `calandcount` kernel. This kernel processes the celestial object data, calculating angular separations and updating a histogram array in parallel.

The key parallelization steps include:
- Dividing the workload among GPU threads with the `blockIdx` and `threadIdx` indices.
- Performing calculations concurrently for different data points, improving overall efficiency.

## Results

The code outputs the results to a file specified by the user. The results include the angular correlation function values (omega) and histograms for real-real (DD), random-random (RR), and real-random (DR) galaxy pairs.

### GPU Nodes and Time Taken

The time taken for kernel execution is printed, providing insights into the efficiency of the parallel implementation. The code dynamically adjusts to the available GPU resources for improved performance. 

> For one NVIDIA GPU node, the program took 6 seconds to finnish.

### Running the Code

To execute the code, use the following command:

```bash
./executable real_data_file random_data_file output_data_file
```

- `real_data_file`: File containing real galaxy data (RA, DEC).
- `random_data_file`: File containing simulated random galaxy data (RA, DEC).
- `output_data_file`: Output file to store the results.

Make sure to provide the correct file paths and names as command-line arguments.

## Dependencies

- CUDA Toolkit
- NVIDIA GPU with CUDA support

## Notes

- Ensure that the input data is provided in arc minutes.
- Verify CUDA device information and adjust GPU settings if needed.
- For optimal performance, consider running the code on a machine with a compatible NVIDIA GPU.

Feel free to contact me at saadwasem@gmail.com for data or any questions related to the project.