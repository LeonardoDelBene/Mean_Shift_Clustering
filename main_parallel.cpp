#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <map>
#include <tuple>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <unordered_map>

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

void mean_shift_clustering(Point* points, Point* shifted_points, int num_points, double bandwidth, int max_iter = 300, double tol = 1e-3) {

    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        Point current_point = points[i];

        bool converged = false; // Variabile per controllare la convergenza
        for (int iter = 0; iter < max_iter && !converged; iter++) {
            Point numerator(0, 0, 0, 0, 0);
            double denominator = 0.0;

            // Calcola lo spostamento
            double numerator_x = 0.0;
            double numerator_y = 0.0;
            double numerator_r = 0.0;
            double numerator_g = 0.0;
            double numerator_b = 0.0;

#pragma omp parallel for reduction(+:numerator_x, numerator_y, numerator_r, numerator_g, numerator_b, denominator)
            for (int j = 0; j < num_points; ++j) {
                double dist = current_point.distance(points[j]);
                double weight = gaussian_kernel(dist, bandwidth);

                // Calcolo locale dei contributi
                double weighted_x = points[j].x * weight;
                double weighted_y = points[j].y * weight;
                double weighted_r = points[j].r * weight;
                double weighted_g = points[j].g * weight;
                double weighted_b = points[j].b * weight;

                // Riduzione sui singoli componenti
                numerator_x += weighted_x;
                numerator_y += weighted_y;
                numerator_r += weighted_r;
                numerator_g += weighted_g;
                numerator_b += weighted_b;
                denominator += weight;
            }

            // Assemblaggio del risultato finale nella struttura
            numerator = Point(numerator_x, numerator_y, numerator_r, numerator_g, numerator_b);
            Point new_point = numerator / denominator;
            double shift_distance = new_point.distance(current_point);

            // Controlla la convergenza
            converged = (shift_distance < tol);

            current_point = new_point;
        }
        shifted_points[i] = current_point;
    }
}

// Funzione principale per la segmentazione dell'immagine
void mean_shift_image_segmentation(unsigned char* image, int width, int height, int channels, double bandwidth, int num_threads) {
    int num_points = width * height;
    Point* points = new Point[num_points];
    Point* shifted_points = new Point[num_points];

    std::cout << "Clustering with Mean-Shift..." << std::endl;
    std::cout << "Number of points: " << num_points << std::endl;

    // Carica i pixel dell'immagine come punti con posizione spaziale e colore
#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        int y_offset = y * width;
        for (int x = 0; x < width; ++x) {
            int index = y_offset + x;
            unsigned char r = image[index * channels + 0];
            unsigned char g = image[index * channels + 1];
            unsigned char b = image[index * channels + 2];
            points[index] = Point(x, y, r, g, b);
        }
    }


    // Applica il clustering Mean-Shift
    mean_shift_clustering(points, shifted_points, num_points, bandwidth);

    std::map<std::tuple<int, int, int, int, int>, int> cluster_map;

    // Parallelize the counting of clusters
#pragma omp parallel
    {
        // Crea una mappa locale per ciascun thread
        std::map<std::tuple<double, double, double, double, double>, int> local_cluster_map;

#pragma omp for
        for (int i = 0; i < num_points; ++i) {
            const Point& point = shifted_points[i];
            std::tuple<double,double,double,double,double> cluster_center = {point.x, point.y, point.r, point.g, point.b};
            local_cluster_map[cluster_center]++;
        }

        // Combinare le mappe locali nel cluster_map globale
#pragma omp critical
        {
            for (const auto& local_cluster : local_cluster_map) {
                cluster_map[local_cluster.first] += local_cluster.second;
            }
        }
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

    // Ricostruisce l'immagine usando il colore medio dei cluster
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        int x = static_cast<int>(points[i].x);
        int y = static_cast<int>(points[i].y);
        image[(y * width + x) * channels + 0] = static_cast<unsigned char>(shifted_points[i].r);  // R
        image[(y * width + x) * channels + 1] = static_cast<unsigned char>(shifted_points[i].g);  // G
        image[(y * width + x) * channels + 2] = static_cast<unsigned char>(shifted_points[i].b);  // B
    }

    delete[] points;
    delete[] shifted_points;
}

int main() {
    int num_threads= 16;
    omp_set_num_threads(num_threads);

    // Carica l'immagine usando stb_image
    int width, height, channels;
    unsigned char* image = stbi_load("img/input.png", &width, &height, &channels, 3);  // Carica come RGB
    if (image == nullptr) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    double bandwidth = 20.0;

    // Esegui la segmentazione dell'immagine con Mean-Shift
    auto start_total = std::chrono::high_resolution_clock::now();
    mean_shift_image_segmentation(image, width, height, channels, bandwidth,num_threads);
    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count() << " seconds" << std::endl;

    // Salva l'immagine segmentata usando stb_image_write
    stbi_write_png("img/output.png", width, height, channels, image, width * channels);

    // Libera la memoria dell'immagine
    stbi_image_free(image);

    return 0;
}

