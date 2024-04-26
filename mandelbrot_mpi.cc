#include <iostream> // Include for input and output stream operations
#include <iomanip>  // For std::setw and std::left
#include <fstream>  // Include for file stream operations
#include <complex>  // Include for complex number operations
#include <vector>   // Include for using the vector container
#include <cstdlib>  // Include for standard library functions, like atoi (ASCII to integer) and atof (ASCII to float)
#include <cmath>    // Include for mathematical functions, like sqrt and sin
#include <string>   // Include for using the string class
#include <omp.h>    // Include for OpenMP parallel programming
#include <mpi.h>    // Include for MPI distributed programming

// Constants defining the output image size and anti-aliasing samples
const int WIDTH = 1920;  // Image width in pixels
const int HEIGHT = 1080; // Image height in pixels

// Forward declarations of functions used in this program
void parseArguments(int argc, char *argv[], int &max_iter, double &center_x, double &center_y, double &zoom, std::string &filename, int &aaSamples);
int computeMandelbrot(double real, double imag, int max_iter);
void mapColor(int iter, int max_iter, int &r, int &g, int &b);

int main(int argc, char *argv[])
{
    int max_iter;
    double center_x, center_y;
    double zoom;
    std::string filename;
    int aaSamples;
    int numNodes, rank;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse command-line arguments
    parseArguments(argc, argv, max_iter, center_x, center_y, zoom, filename, aaSamples);
    int aaSide = std::sqrt(aaSamples);

    // Broadcast input parameters to all nodes
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&center_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zoom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&aaSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate scale and movement factors based on zoom and center coordinates
    double scale = 4.0 / (WIDTH * zoom);
    double move_x = center_x - WIDTH / 2.0 * scale;
    double move_y = center_y - HEIGHT / 2.0 * scale;

    // Calculate the range of rows assigned to each node
    int chunkSize = HEIGHT / numNodes;
    int rowStart = rank * chunkSize;
    int rowEnd = (rank == numNodes - 1) ? HEIGHT : (rank + 1) * chunkSize;

    // Create a local buffer to store the computed image data for each node
    std::vector<unsigned char> localData(3 * WIDTH * (rowEnd - rowStart));

    // Synchronize all nodes before starting the computation
    MPI_Barrier(MPI_COMM_WORLD);
    double compStart = MPI_Wtime();

// Compute the Mandelbrot set in parallel using OpenMP
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = rowStart; y < rowEnd; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            double totalR = 0, totalG = 0, totalB = 0;
            for (int dy = 0; dy < aaSide; ++dy)
            {
                for (int dx = 0; dx < aaSide; ++dx)
                {
                    // Calculate the real and imaginary parts of the complex number
                    double real = (x + (dx / (double)aaSide)) * scale + move_x;
                    double imag = (y + (dy / (double)aaSide)) * scale + move_y;
                    // Compute the number of iterations for the current pixel
                    int iter = computeMandelbrot(real, imag, max_iter);
                    // Map the iteration count to RGB color values
                    int r, g, b;
                    mapColor(iter, max_iter, r, g, b);
                    // Accumulate the color values for anti-aliasing
                    totalR += r;
                    totalG += g;
                    totalB += b;
                }
            }
            // Calculate the average color values and store them in the local buffer
            int idx = 3 * ((y - rowStart) * WIDTH + x);
            localData[idx] = std::min(255, static_cast<int>(totalR / aaSamples));
            localData[idx + 1] = std::min(255, static_cast<int>(totalG / aaSamples));
            localData[idx + 2] = std::min(255, static_cast<int>(totalB / aaSamples));
        }
    }

    // Synchronize all nodes after the computation
    MPI_Barrier(MPI_COMM_WORLD);
    double compEnd = MPI_Wtime();
    double compTime = compEnd - compStart;

    // Reduce the maximum computation time across all nodes
    double maxCompTime;
    MPI_Reduce(&compTime, &maxCompTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Create a buffer to store the final image data on the root node
    std::vector<unsigned char> data(3 * WIDTH * HEIGHT);

    // Prepare the receive counts and displacements for MPI_Gatherv
    int *recvCounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        recvCounts = new int[numNodes];
        displs = new int[numNodes];
        for (int i = 0; i < numNodes; ++i)
        {
            recvCounts[i] = 3 * WIDTH * (HEIGHT / numNodes);
            displs[i] = 3 * i * WIDTH * (HEIGHT / numNodes);
        }
        recvCounts[numNodes - 1] = 3 * WIDTH * (HEIGHT - (numNodes - 1) * (HEIGHT / numNodes));
    }

    // Gather the local image data from all nodes to the root node
    MPI_Gatherv(localData.data(), 3 * WIDTH * (rowEnd - rowStart), MPI_UNSIGNED_CHAR,
                data.data(), recvCounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Write the final image to a file and measure the I/O time on the root node
    double ioStart, ioEnd, ioTime;
    if (rank == 0)
    {
        delete[] recvCounts;
        delete[] displs;

        ioStart = MPI_Wtime();

        std::ofstream imageFile(filename, std::ios::binary);
        imageFile << "P6\n"
                  << WIDTH << " " << HEIGHT << "\n255\n";
        imageFile.write(reinterpret_cast<char *>(data.data()), 3 * WIDTH * HEIGHT);
        imageFile.close();

        ioEnd = MPI_Wtime();
        ioTime = ioEnd - ioStart;

        // Print the computation time and file write time
        std::cout << "Computation Time: " << maxCompTime << " seconds\n";
        std::cout << "File Write Time: " << ioTime << " seconds\n";
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}

// Function to parse command-line arguments
void parseArguments(int argc, char *argv[], int &max_iter, double &center_x, double &center_y, double &zoom, std::string &filename, int &aaSamples)
{
    // Set default values for the parameters
    aaSamples = 4;
    filename = "mandelbrot.pnm";
    max_iter = 10000;
    center_x = -0.75;
    center_y = 0.0;
    zoom = 1.0;

    // Loop through the command-line arguments to override defaults
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc)
        {
            filename = argv[++i];
            if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".pnm")
            {
                filename += ".pnm";
            }
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            max_iter = std::stoi(argv[++i]);
        }
        else if (arg == "-x" && i + 1 < argc)
        {
            center_x = atof(argv[++i]);
        }
        else if (arg == "-y" && i + 1 < argc)
        {
            center_y = atof(argv[++i]);
        }
        else if (arg == "-z" && i + 1 < argc)
        {
            zoom = atof(argv[++i]);
        }
        else if (arg == "-aa" && i + 1 < argc)
        {
            aaSamples = std::stoi(argv[++i]);
            if (aaSamples < 1)
                aaSamples = 1;
        }
    }

    // Print a summary of the conditions being used for this run
    std::cout << "\n=== Mandelbrot Set Generation Conditions ===\n";
    std::cout << std::left << std::setw(20) << "Output Filename:" << filename << "\n";
    std::cout << std::left << std::setw(20) << "Max Iterations:" << max_iter << "\n";
    std::cout << std::left << std::setw(20) << "Center X:" << center_x << "\n";
    std::cout << std::left << std::setw(20) << "Center Y:" << center_y << "\n";
    std::cout << std::left << std::setw(20) << "Zoom Level:" << zoom << "\n";
    std::cout << std::left << std::setw(20) << "AA Samples:" << aaSamples << "\n";
    std::cout << "============================================\n";
}

// Function to compute the number of iterations for a complex number to escape the Mandelbrot set
int computeMandelbrot(double real, double imag, int max_iter)
{
    std::complex<double> c(real, imag);
    std::complex<double> z(0, 0);
    int n = 0;
    while (abs(z) <= 2.0 && n < max_iter)
    {
        z = z * z + c;
        ++n;
    }
    return n;
}

// Function to map the iteration count to RGB color values
void mapColor(int iter, int max_iter, int &r, int &g, int &b)
{
    if (iter == max_iter)
    {
        r = g = b = 0;
    }
    else
    {
        double frequency = 0.1;
        r = static_cast<int>(sin(frequency * iter + 0) * 127 + 128);
        g = static_cast<int>(sin(frequency * iter + 2) * 127 + 128);
        b = static_cast<int>(sin(frequency * iter + 4) * 127 + 128);
    }
}