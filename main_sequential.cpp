#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
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

void mean_shift_clustering(Point* points, int num_points, double bandwidth, int max_iter, double tol, Point* shifted_points) {
    // Perform Mean-Shift for each point
    for (int i = 0; i < num_points; ++i) {
        Point current_point = points[i];

        // Perform Mean-Shift for the current point
        for (int iter = 0; iter < max_iter; ++iter) {
            Point numerator(0, 0, 0, 0, 0);
            double denominator = 0.0;

            // Calculate the shift for the current point
            for (int j = 0; j < num_points; ++j) {
                double dist = current_point.distance(points[j]);
                double weight = gaussian_kernel(dist, bandwidth);
                numerator = numerator + points[j] * weight;
                denominator += weight;
            }
            Point new_point = numerator / denominator;
            double shift_distance = new_point.distance(current_point);

            if (shift_distance < tol)
                break;

            current_point = new_point;
        }

        // Store the shifted point
        shifted_points[i] = current_point;
    }
}

void mean_shift_image_segmentation(unsigned char* image, int width, int height, int channels, double bandwidth) {
    int num_points = width * height;
    Point* points = new Point[num_points];

    std::cout << "Clustering with Mean-Shift..." << std::endl;
    std::cout << "Number of points: " << num_points << std::endl;

    // Load the image pixels as points with color and spatial position
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char r = image[(y * width + x) * channels + 0];
            unsigned char g = image[(y * width + x) * channels + 1];
            unsigned char b = image[(y * width + x) * channels + 2];
            points[(y * width + x)] = Point(x, y, r, g, b);
        }
    }

    // Perform clustering
    Point* shifted_points = new Point[num_points];
    int max_iter = 300;
    double tol = 1e-3;

    // Call the mean shift clustering function
    mean_shift_clustering(points, num_points, bandwidth, max_iter, tol, shifted_points);

    // Map to store clusters with their centroid (x, y, r, g, b) and population count
    std::map<std::tuple<int, int, int, int, int>, int> cluster_map;

    for (int i = 0; i < num_points; ++i) {
        std::tuple<int, int, int, int, int> cluster_center = {static_cast<int>(shifted_points[i].x), static_cast<int>(shifted_points[i].y),
                                                              static_cast<int>(shifted_points[i].r), static_cast<int>(shifted_points[i].g),
                                                              static_cast<int>(shifted_points[i].b)};
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
    for (int i = 0; i < num_points; ++i) {
        int x = static_cast<int>(std::round(points[i].x));
        int y = static_cast<int>(std::round(points[i].y));

        image[(y * width + x) * channels + 0] = static_cast<unsigned char>(shifted_points[i].r);  // R
        image[(y * width + x) * channels + 1] = static_cast<unsigned char>(shifted_points[i].g);  // G
        image[(y * width + x) * channels + 2] = static_cast<unsigned char>(shifted_points[i].b);  // B
    }

    delete[] points;
    delete[] shifted_points;
}

int main() {
    // Load the image using stb_image
    int width, height, channels;
    unsigned char* original_image = stbi_load("img/input 2.png", &width, &height, &channels, 0); // Carica tutti i canali
    if (channels == 4) {
        unsigned char* rgb_image = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height; ++i) {
            rgb_image[i * 3 + 0] = original_image[i * 4 + 0]; // R
            rgb_image[i * 3 + 1] = original_image[i * 4 + 1]; // G
            rgb_image[i * 3 + 2] = original_image[i * 4 + 2]; // B
        }
        stbi_image_free(original_image);
        original_image = rgb_image;
        channels = 3;
    }

    double bandwidth = 20.0;

    // Perform the image segmentation with Mean-Shift
    auto start_total = std::chrono::high_resolution_clock::now();
    mean_shift_image_segmentation(original_image, width, height, channels, bandwidth);
    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count() << " milliseconds" << std::endl;

    // Save the segmented image using stb_image_write
    stbi_write_png("img/output 2.png", width, height, channels, original_image, width * 3);

    stbi_image_free(original_image);
    return 0;
}




