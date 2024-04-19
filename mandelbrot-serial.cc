#include <iostream> // Include for input and output stream operations
#include <iomanip> // For std::setw and std::left
#include <fstream> // Include for file stream operations
#include <complex> // Include for complex number operations
#include <vector> // Include for using the vector container
#include <cstdlib> // Include for standard library functions, like atoi (ASCII to integer) and atof (ASCII to float)
#include <cmath> // Include for mathematical functions, like sqrt and sin
#include <string> // Include for using the string class

// Constants defining the output image size and anti-aliasing samples
const int WIDTH = 1920; // Image width in pixels
const int HEIGHT = 1080; // Image height in pixels

// Forward declarations of functions used in this program
void parseArguments(int argc, char *argv[], int &max_iter, double &center_x, double &center_y, double &zoom, std::string &filename, int &aaSamples);
int computeMandelbrot(double real, double imag, int max_iter);
void mapColor(int iter, int max_iter, int &r, int &g, int &b);

int main(int argc, char* argv[]) {
    // Variables to hold the parameters for generating the Mandelbrot set image
    int max_iter; // Maximum iterations for determining if a point is in the Mandelbrot set
    double center_x, center_y; // Center coordinates of the view
    double zoom; // Zoom level
    std::string filename; // Output filename for the image

    int aaSamples; // Variable to hold the number of anti-aliasing samples per pixel
    // Parse command-line arguments to set the above parameters
    parseArguments(argc, argv, max_iter, center_x, center_y, zoom, filename, aaSamples);

    // Calculate the side length of the anti-aliasing square grid
    int aaSide = std::sqrt(aaSamples);

    // Compute scale factors for the Mandelbrot set based on the zoom level and image dimensions
    double scale = 4.0 / (WIDTH * zoom);
    double move_x = center_x - WIDTH / 2.0 * scale;
    double move_y = center_y - HEIGHT / 2.0 * scale;

    // Vectors to store the red, green, and blue components of each pixel
    std::vector<int> red(WIDTH * HEIGHT);
    std::vector<int> green(WIDTH * HEIGHT);
    std::vector<int> blue(WIDTH * HEIGHT);

    // Generate the image
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            // Variables to accumulate color values for anti-aliasing
            double totalR = 0, totalG = 0, totalB = 0;
            for (int dy = 0; dy < aaSide; ++dy) {
                for (int dx = 0; dx < aaSide; ++dx) {
                    // Compute the real and imaginary parts of the complex number for this sample
                    double real = (x + (dx / (double)aaSide)) * scale + move_x;
                    double imag = (y + (dy / (double)aaSide)) * scale + move_y;
                    // Compute how many iterations it takes for the complex number to escape
                    int iter = computeMandelbrot(real, imag, max_iter);
                    // Map the iteration count to a color
                    int r, g, b;
                    mapColor(iter, max_iter, r, g, b);
                    // Accumulate the color values
                    totalR += r;
                    totalG += g;
                    totalB += b;
                }
            }
            // Compute the average color values for this pixel and clamp to [0, 255]
            int idx = y * WIDTH + x;
            red[idx] = std::min(255, static_cast<int>(totalR / aaSamples));
            green[idx] = std::min(255, static_cast<int>(totalG / aaSamples));
            blue[idx] = std::min(255, static_cast<int>(totalB / aaSamples));
        }
    }

    // Open the output file
    std::ofstream imageFile(filename);
    // Write the PNM file header
    imageFile << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    // Write the pixel data
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int idx = y * WIDTH + x;
            imageFile << red[idx] << " " << green[idx] << " " << blue[idx] << "\n";
        }
    }
    // Close the file
    imageFile.close();

    return 0; // Successful program termination
}

void parseArguments(int argc, char *argv[], int &max_iter, double &center_x, double &center_y, double &zoom, std::string &filename, int &aaSamples) {
    // Default values for the parameters
    aaSamples = 4; // Default AA samples to 4 (2x2 grid)
    filename = "mandelbrot.pnm"; // Default output filename
    max_iter = 10000; // Default maximum iterations for the Mandelbrot computation
    center_x = -0.75; // Default X coordinate of the view center
    center_y = 0.0; // Default Y coordinate of the view center
    zoom = 1.0; // Default zoom level

    // Loop through the command-line arguments to override defaults
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
            if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".pnm") {
                filename += ".pnm";
            }
        } else if (arg == "-i" && i + 1 < argc) {
            max_iter = std::stoi(argv[++i]);
        } else if (arg == "-x" && i + 1 < argc) {
            center_x = atof(argv[++i]);
        } else if (arg == "-y" && i + 1 < argc) {
            center_y = atof(argv[++i]);
        } else if (arg == "-z" && i + 1 < argc) {
            zoom = atof(argv[++i]);
        } else if (arg == "-aa" && i + 1 < argc) {
            aaSamples = std::stoi(argv[++i]);
            if (aaSamples < 1) aaSamples = 1;
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


// This function computes the number of iterations it takes for a complex number to escape the Mandelbrot set.
int computeMandelbrot(double real, double imag, int max_iter) {
    std::complex<double> c(real, imag); // The complex number for which we compute the Mandelbrot iteration
    std::complex<double> z(0, 0); // The initial value of z in the Mandelbrot iteration
    int n = 0; // Iteration counter
    // Iterate until |z| > 2 (escaped) or we reach the maximum number of iterations
    while (abs(z) <= 2.0 && n < max_iter) {
        z = z*z + c;
        ++n;
    }
    return n; // Return the number of iterations
}

// This function maps an iteration count to a color using a sinusoidal function.
void mapColor(int iter, int max_iter, int &r, int &g, int &b) {
    if (iter == max_iter) {
        // If the point is in the Mandelbrot set, color it black
        r = g = b = 0;
    } else {
        // Otherwise, compute a color based on the iteration count
        double frequency = 0.1;
        r = static_cast<int>(sin(frequency * iter + 0) * 127 + 128);
        g = static_cast<int>(sin(frequency * iter + 2) * 127 + 128);
        b = static_cast<int>(sin(frequency * iter + 4) * 127 + 128);
    }
}
