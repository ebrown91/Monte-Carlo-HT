#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

const int WIDTH = 100;
const int HEIGHT = 100;
const int NUM_PARTICLES = 5000;
const int STEPS = 500;
const double ENERGY_PER_PARTICLE = 1.0;
const double SCALE_FACTOR = 6.0; // Display scale

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

// Create a vertical temperature scale bar
cv::Mat createTemperatureScale(double minVal, double maxVal, int height) {
    cv::Mat scale(height, 50, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        scale.at<uchar>(y, 0) = static_cast<uchar>(255.0 * (1.0 - (double)y / height));
        for (int x = 1; x < 50; ++x) {
            scale.at<uchar>(y, x) = scale.at<uchar>(y, 0);
        }
    }
    cv::Mat colorScale;
    cv::applyColorMap(scale, colorScale, cv::COLORMAP_JET);

    // Add text labels for min and max
    cv::putText(colorScale, cv::format("%.2f", maxVal), cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(colorScale, cv::format("%.2f", minVal), cv::Point(5, height - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    return colorScale;
}

int main() {
    try {
        std::vector<std::vector<double>> temperature(HEIGHT, std::vector<double>(WIDTH, 0.0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distX(0, WIDTH - 1);
        std::uniform_int_distribution<> distY(0, HEIGHT - 1);
        std::uniform_int_distribution<> moveDir(0, 3);

        std::vector<std::pair<int, int>> particles(NUM_PARTICLES);
        for (auto& p : particles) {
            p.first = distX(gen);
            p.second = distY(gen);
        }

        cv::namedWindow("Monte Carlo Heat Transfer", cv::WINDOW_NORMAL);

        for (int step = 0; step < STEPS; ++step) {
            for (auto& p : particles) {
                temperature[p.second][p.first] += ENERGY_PER_PARTICLE / STEPS;

                int dir = moveDir(gen);
                if (dir == 0 && p.second > 0) p.second--;
                else if (dir == 1 && p.second < HEIGHT - 1) p.second++;
                else if (dir == 2 && p.first > 0) p.first--;
                else if (dir == 3 && p.first < WIDTH - 1) p.first++;
            }

            double minVal, maxVal;
            cv::Mat gray = normalizeGrid(temperature, minVal, maxVal);

            // Apply color map
            cv::Mat colorImg;
            cv::applyColorMap(gray, colorImg, cv::COLORMAP_JET);

            // Scale up simulation
            cv::Mat scaledSim;
            cv::resize(colorImg, scaledSim, cv::Size(), SCALE_FACTOR, SCALE_FACTOR, cv::INTER_NEAREST);

            // Create and scale temperature bar
            cv::Mat scaleBar = createTemperatureScale(minVal, maxVal, scaledSim.rows);

            // Combine simulation and scale bar
            cv::Mat display;
            cv::hconcat(scaledSim, scaleBar, display);

            cv::imshow("Monte Carlo Heat Transfer", display);

            if (cv::waitKey(1) == 27) break; // ESC to exit
        }

        cv::waitKey(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
