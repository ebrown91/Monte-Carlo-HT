#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <opencv2/opencv.hpp>

const int WIDTH = 100;
const int HEIGHT = 100;
const int NUM_PARTICLES = 5000;
const int STEPS = 500;
const double ENERGY_PER_PARTICLE = 1.0;

// Convert temperature grid to grayscale image
cv::Mat normalizeToHeatmap(const std::vector<std::vector<double>>& grid) {
    double minVal = grid[0][0], maxVal = grid[0][0];
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

        cv::namedWindow("Monte Carlo Heat Transfer", cv::WINDOW_AUTOSIZE);

        for (int step = 0; step < STEPS; ++step) {
            for (auto& p : particles) {
                temperature[p.second][p.first] += ENERGY_PER_PARTICLE / STEPS;
                int dir = moveDir(gen);
                if (dir == 0 && p.second > 0) p.second--;
                else if (dir == 1 && p.second < HEIGHT - 1) p.second++;
                else if (dir == 2 && p.first > 0) p.first--;
                else if (dir == 3 && p.first < WIDTH - 1) p.first++;
            }
            cv::Mat grayImg = normalizeToHeatmap(temperature);
            cv::Mat colorImg;
            cv::applyColorMap(grayImg, colorImg, cv::COLORMAP_JET);
            cv::imshow("Monte Carlo Heat Transfer", colorImg);
            if (cv::waitKey(10) == 27) break; // ESC to exit
        }
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
