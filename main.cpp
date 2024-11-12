#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <chrono>
#include <cmath>

struct Point {
    double x, y, r, g, b;

    Point(double x, double y, double r, double g, double b) : x(x), y(y), r(r), g(g), b(b) {}

    Point() : x(0), y(0), r(0), g(0), b(0) {}

    Point operator+(const Point &p) const {
        return Point(x + p.x, y + p.y, r + p.r, g + p.g, b + p.b);
    }

    Point operator*(double a) const {
        return Point(x * a, y * a, r * a, g * a, b * a);
    }

    Point operator/(double a) const {
        return Point(x / a, y / a, r / a, g / a, b / a);
    }

    double distance(const Point &p) const {
        return std::sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (r - p.r) * (r - p.r) + (g - p.g) * (g - p.g) + (b - p.b) * (b - p.b));
    }
};

double gaussian_kernel(double distance, double bandwidth) {
    return std::exp(-0.5 * (distance * distance) / (bandwidth * bandwidth));
}

Point mean_shift_single_point(const Point& point, const std::vector<Point>& points, double bandwidth, int max_iter, double tol) {
    Point current_point = point;

    for (int i = 0; i < max_iter; i++) {
        Point numerator(0, 0, 0, 0, 0);
        double denominator = 0.0;

        // Calculate the shift
        for (const Point& p : points) {
            double dist = current_point.distance(p);
            double weight = gaussian_kernel(dist, bandwidth);
            numerator = numerator + p * weight;
            denominator += weight;
        }
        Point new_point = numerator / denominator;
        double shift_distance = new_point.distance(current_point);

        if (shift_distance < tol)
            break;

        current_point = new_point;
    }
    return current_point;
}

std::vector<Point> mean_shift_clustering(const std::vector<Point>& points, double bandwidth, int max_iter = 300, double tol = 1e-3) {
    std::vector<Point> shifted_points(points.size());
    std::cout << "Clustering with Mean-Shift..." << std::endl;
    std::cout << "Number of points: " << points.size() << std::endl;

    // Perform mean-shift on each point
    for (int i = 0; i < points.size(); ++i) {
        shifted_points[i] = mean_shift_single_point(points[i], points, bandwidth, max_iter, tol);
        std::cout << "Point " << i + 1 << " / " << points.size() << std::endl;
    }

    return shifted_points;
}

// Main function to segment the image
void mean_shift_image_segmentation(unsigned char* image, int width, int height, int channels, double bandwidth) {
    std::vector<Point> points;

    // Load the image pixels as points with color and spatial position
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char r = image[(y * width + x) * channels + 0];
            unsigned char g = image[(y * width + x) * channels + 1];
            unsigned char b = image[(y * width + x) * channels + 2];
            points.emplace_back(x, y, r, g, b);
        }
    }

    // Apply Mean-Shift Clustering
    std::vector<Point> shifted_points = mean_shift_clustering(points, bandwidth);

    // Map to store clusters with their centroid (x, y, r, g, b) and population count
    std::map<std::tuple<int, int, int, int, int>, int> cluster_map;

    for (const auto& point : shifted_points) {
        std::tuple<int, int, int, int, int> cluster_center = {static_cast<int>(point.x), static_cast<int>(point.y),
                                                              static_cast<int>(point.r), static_cast<int>(point.g),
                                                              static_cast<int>(point.b)};
        cluster_map[cluster_center]++;
    }

    std::cout << "Total clusters: " << cluster_map.size() << std::endl;
    int cluster_id = 1;
    for (const auto& cluster : cluster_map) {
        std::cout << "Cluster " << cluster_id << " (X:" << std::get<0>(cluster.first)
                  << " Y:" << std::get<1>(cluster.first)
                  << " R:" << std::get<2>(cluster.first)
                  << " G:" << std::get<3>(cluster.first)
                  << " B:" << std::get<4>(cluster.first) << ") -> "
                  << "Population: " << cluster.second << std::endl;
        cluster_id++;
    }

    // Rebuild the image using the mean color of the clusters
    for (int i = 0; i < points.size(); ++i) {
        int x = static_cast<int>(points[i].x);
        int y = static_cast<int>(points[i].y);
        image[(y * width + x) * channels + 0] = static_cast<unsigned char>(shifted_points[i].r);  // R
        image[(y * width + x) * channels + 1] = static_cast<unsigned char>(shifted_points[i].g);  // G
        image[(y * width + x) * channels + 2] = static_cast<unsigned char>(shifted_points[i].b);  // B
    }
}

int main() {
    // Load the image using stb_image
    int width, height, channels;
    unsigned char* image = stbi_load("img/input.png", &width, &height, &channels, 3);  // Load as RGB
    if (image == nullptr) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    double bandwidth = 20.0;

    // Perform the image segmentation with Mean-Shift
    auto start_total = std::chrono::high_resolution_clock::now();
    mean_shift_image_segmentation(image, width, height, channels, bandwidth);
    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total).count() << " seconds" << std::endl;

    // Save the segmented image using stb_image_write
    stbi_write_png("img/output.png", width, height, channels, image, width * channels);

    // Free the image memory
    stbi_image_free(image);

    return 0;
}



