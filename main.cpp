#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

// Grid size
const int WIDTH = 100;
const int HEIGHT = 100;

// Display
const double SCALE_FACTOR = 6.0;

// Dirichlet boundary values (left/right walls, fixed for all time)
const double LEFT_TEMP = 100.0;
const double RIGHT_TEMP = 0.0;

// Monte Carlo relaxation parameters
const int SAMPLES_PER_CELL = 20;   // random neighbor samples per cell per iteration
const double TOLERANCE = 1e-3;     // convergence threshold
const int MAX_ITERATIONS = 5000;   // safety cap

// Convert temperature grid to normalized grayscale
cv::Mat normalizeGrid(const std::vector<std::vector<double>>& grid, double& minVal, double& maxVal) {
    minVal = grid[0][0];
    maxVal = grid[0][0];
    for (const auto& row : grid) {
        for (double val : row) {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }

    cv::Mat img(HEIGHT, WIDTH, CV_8UC1);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            img.at<uchar>(y, x) = static_cast<uchar>(
                255.0 * (grid[y][x] - minVal) / (maxVal - minVal + 1e-9)
            );
        }
    }
    return img;
}

cv::Mat createTemperatureScale(double minVal, double maxVal, int height) {
    cv::Mat scale(height, 50, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        uchar value = static_cast<uchar>(255.0 * (1.0 - (double)y / height));
        for (int x = 0; x < 50; ++x) {
            scale.at<uchar>(y, x) = value;
        }
    }
    cv::Mat colorScale;
    cv::applyColorMap(scale, colorScale, cv::COLORMAP_JET);

    cv::putText(colorScale, cv::format("%.2f", maxVal), cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(colorScale, cv::format("%.2f", minVal), cv::Point(5, height - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    return colorScale;
}

void renderFrame(const std::vector<std::vector<double>>& temperature, int iteration) {
    double minVal, maxVal;
    cv::Mat gray = normalizeGrid(temperature, minVal, maxVal);

    cv::Mat colorImg;
    cv::applyColorMap(gray, colorImg, cv::COLORMAP_JET);

    cv::Mat resized;
    cv::resize(colorImg, resized,
               cv::Size(static_cast<int>(WIDTH * SCALE_FACTOR), static_cast<int>(HEIGHT * SCALE_FACTOR)),
               0, 0, cv::INTER_NEAREST);

    cv::Mat scaleBar = createTemperatureScale(minVal, maxVal, resized.rows);
    cv::Mat display;
    cv::hconcat(resized, scaleBar, display);

    cv::putText(display, cv::format("Iteration: %d", iteration), cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    cv::imshow("Monte Carlo Heat Transfer", display);
    cv::waitKey(1);
}

int main() {
    try {
        std::vector<std::vector<double>> temperature(HEIGHT, std::vector<double>(WIDTH, 0.0));

        // ---- Set initial boundary conditions ----
        for (int y = 0; y < HEIGHT; ++y) {
            temperature[y][0] = LEFT_TEMP;
            temperature[y][WIDTH - 1] = RIGHT_TEMP;
        }
        // Interior starts at 0.0 (already initialized above); top/bottom are
        // insulated (no fixed value), so heat only enters from left/right.

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> neighborDir(0, 3); // 0=up,1=down,2=left,3=right

        cv::namedWindow("Monte Carlo Heat Transfer", cv::WINDOW_NORMAL);

        int iteration = 0;
        bool converged = false;

        while (!converged && iteration < MAX_ITERATIONS) {
            iteration++;
            std::vector<std::vector<double>> newTemp = temperature; // Jacobi: read from old, write to new
            double maxChange = 0.0;

            for (int y = 0; y < HEIGHT; ++y) {
                for (int x = 1; x < WIDTH - 1; ++x) { // skip fixed left/right walls
                    double sum = 0.0;

                    for (int s = 0; s < SAMPLES_PER_CELL; ++s) {
                        int nx = x, ny = y;
                        int dir = neighborDir(gen);
                        if (dir == 0) ny--;
                        else if (dir == 1) ny++;
                        else if (dir == 2) nx--;
                        else if (dir == 3) nx++;

                        // Insulated top/bottom: reflect back into grid
                        if (ny < 0) ny = 1;
                        if (ny >= HEIGHT) ny = HEIGHT - 2;
                        // x is always in-range here since x starts at 1 and
                        // ends at WIDTH-2, so nx is at worst 0 or WIDTH-1 —
                        // exactly the fixed Dirichlet walls, which is correct.

                        sum += temperature[ny][nx];
                    }

                    newTemp[y][x] = sum / SAMPLES_PER_CELL;
                    double diff = std::abs(newTemp[y][x] - temperature[y][x]);
                    if (diff > maxChange) maxChange = diff;
                }
            }

            temperature = newTemp;
            if (maxChange < TOLERANCE) converged = true;

            if (iteration % 10 == 0 || converged) {
                renderFrame(temperature, iteration);
            }
        }

        std::cout << (converged ? "Converged" : "Stopped (max iterations reached)")
                  << " after " << iteration << " iterations." << std::endl;
        std::cout << "Press any key on the image window to exit..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();

        return 0;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}