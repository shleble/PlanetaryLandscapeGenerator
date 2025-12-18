#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include "../Dependencies/FastNoiseLite.h"
#include <array>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>

// Vector structure for 3D coordinates
struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};

// Normalize a vector
Vec3 normalize(const Vec3& v) {
    double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len > 0 ? Vec3(v.x / len, v.y / len, v.z / len) : v;
}

// Planetary parameters structure
struct PlanetParams {
    double g;     // Gravitational acceleration (m/s^2)
    double R;     // Radius (km)
    double T;     // Lithosphere thickness (km)
    bool W;       // Water presence
    double A;     // Age (years)
    int S;        // Seed
    double T_min; // Minimum temperature (°C)
    double T_max; // Maximum temperature (°C)
};

// Type alias for heightmap
using Heightmap = std::vector<std::vector<double>>;

// Structure for river network
struct Cell {
    int i, j;
    Cell(int i = -1, int j = -1) : i(i), j(j) {}
    bool operator==(const Cell& other) const { return i == other.i && j == other.j; }
};

// Structure for face and ST coordinates
struct FaceST {
    int face;
    double s, t;
};

class PlanetaryLandscape {
private:
    openvdb::FloatGrid::Ptr grid;
    PlanetParams params;
    static const int N = 256; // Resolution per face
    std::array<Heightmap, 6> z0;        // Initial heightmaps
    std::array<Heightmap, 6> u;         // Uplift maps
    std::array<Heightmap, 6> heightmaps; // Final eroded heightmaps
    FastNoiseLite noise;
    std::mt19937 rng; // Seeded RNG

public:
    const openvdb::FloatGrid::Ptr& getGrid() const { return grid; }
    const std::array<Heightmap, 6>& getHeightmaps() const { return heightmaps; }
    static int getResolution() { return N; }
    PlanetaryLandscape(PlanetParams p) : params(p), rng(p.S) {
        openvdb::initialize();
        grid = openvdb::FloatGrid::create();
        noise.SetSeed(params.S);
        noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        
        // Improve noise for planetary scale
        noise.SetFractalType(FastNoiseLite::FractalType_FBm);
        noise.SetFractalOctaves(5);
        // Dynamic Frequency: 10 cycles per planet radius (approx 20 feature blobs around equator)
        noise.SetFrequency(4.0f / params.R);

        for (auto& hm : z0) hm.resize(N, std::vector<double>(N, 0.0));
        for (auto& hm : u) hm.resize(N, std::vector<double>(N, 0.0));
        for (auto& hm : heightmaps) hm.resize(N, std::vector<double>(N, 0.0));
    }

    // Get direction vector for cube face
    Vec3 get_direction(int face, double s, double t) {
        switch (face) {
            case 0: return normalize(Vec3(1.0, s, t));  // +X
            case 1: return normalize(Vec3(-1.0, s, t)); // -X
            case 2: return normalize(Vec3(s, 1.0, t));  // +Y
            case 3: return normalize(Vec3(s, -1.0, t)); // -Y
            case 4: return normalize(Vec3(s, t, 1.0));  // +Z
            case 5: return normalize(Vec3(s, t, -1.0)); // -Z
            default: return Vec3(0, 0, 0);
        }
    }

    // Generate initial heightmaps and uplift maps
    void generateInitialMaps() {
        double scale = params.R; // Scale coordinates by planet radius (km)
        // Dynamic Amplitude: Scale height with radius to maintain "toy planet" proportions
        double amplitude_z0 = params.R * 0.15; // 15% of radius (e.g., 15km for R=100)
        double amplitude_u = params.R * 0.05;  // 5% uplift base
        for (int face = 0; face < 6; ++face) {
            for (int i = 0; i < N; ++i) {
                double s = -1.0 + 2.0 * i / (N - 1);
                for (int j = 0; j < N; ++j) {
                    double t = -1.0 + 2.0 * j / (N - 1);
                    Vec3 D = get_direction(face, s, t);
                    
                    // Use scaled coordinates with offset
                    double x = D.x * scale;
                    double y = D.y * scale;
                    double z = D.z * scale;

                    z0[face][i][j] = noise.GetNoise(x, y, z) * amplitude_z0;
                    
                    // For uplift, use a different offset and remap to 0..1 to avoid large flat 0 areas
                    double u_noise = noise.GetNoise(x + 2000.0, y + 2000.0, z + 2000.0);
                    u[face][i][j] = (u_noise * 0.5 + 0.5) * amplitude_u; 
                }
            }
        }
    }

    // Compute river network for a face
    std::vector<std::vector<Cell>> computeRiverNetwork(const Heightmap& z, int face) {
        std::vector<std::vector<Cell>> receivers(N, std::vector<Cell>(N, Cell(-1, -1)));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double min_height = z[i][j];
                std::vector<Cell> candidates;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                            if (z[ni][nj] < min_height) {
                                min_height = z[ni][nj];
                                candidates.clear();
                                candidates.push_back({ni, nj});
                            } else if (z[ni][nj] == min_height) {
                                candidates.push_back({ni, nj});
                            }
                        }
                    }
                }
                if (!candidates.empty()) {
                    std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
                    receivers[i][j] = candidates[dist(rng)];
                }
            }
        }
        return receivers;
    }

    // Compute drainage area
    Heightmap computeDrainageArea(const std::vector<std::vector<Cell>>& rn) {
        Heightmap A(N, std::vector<double>(N, 1.0));
        std::vector<std::vector<int>> upstream(N, std::vector<int>(N, 0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Cell r = rn[i][j];
                if (r.i != -1) upstream[r.i][r.j]++;
            }
        }
        std::queue<Cell> q;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (upstream[i][j] == 0 && rn[i][j].i != -1) q.push({i, j});
            }
        }
        while (!q.empty()) {
            Cell c = q.front(); q.pop();
            Cell r = rn[c.i][c.j];
            if (r.i != -1) {
                A[r.i][r.j] += A[c.i][c.j];
                if (--upstream[r.i][r.j] == 0) q.push(r);
            }
        }
        return A;
    }

    // Simplified analytical erosion
    void applyAnalyticalErosion(int face) {
        Heightmap& z = heightmaps[face];
        z = z0[face];
        if (!params.W) {
            double t = params.A / 1e9;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    z[i][j] += u[face][i][j] * t;
                }
            }
            return;
        }
        auto rn = computeRiverNetwork(z, face);
        auto A = computeDrainageArea(rn);
        double temperature_factor = 1.0;
        if (params.T_min < 0.0 && params.T_max > 0.0) {
            double range = params.T_max - params.T_min;
            temperature_factor = 1.0 + (range / 100.0);
        }
        double k_base = 0.005 * (params.g / 9.81); // Decreased from 0.1 to 0.02 to allow steeper slopes
        double k = k_base * temperature_factor;
        double m = 0.4, n = 1.0;
        double total_time = params.A / 1e9;
        double dx = 2.0 / (N - 1);
        
        // Stability fix: ensure dt is small enough relative to dx
        // Target dt = 0.002 ensures stability for N=256 and likely N=512
        double target_dt = dx / 1.9;
        int iterations = static_cast<int>(std::ceil(total_time / target_dt));
        double dt = total_time / iterations;
        
        std::cout << "Erosion iterations: " << iterations << ", dt: " << dt << std::endl;
        
        for (int iter = 0; iter < iterations; ++iter) {
            Heightmap z_new = z;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    Cell r = rn[i][j];
                    if (r.i != -1) {
                        double slope = (z[i][j] - z[r.i][r.j]) / dx;
                        if (slope > 0) {
                            double erosion = k * std::pow(A[i][j], m) * std::pow(slope, n) * dt;
                            // Stability clamp: Don't erode more than the height difference (overshoot check)
                            double max_erosion = z[i][j] - z[r.i][r.j];
                            erosion = std::min(erosion, max_erosion * 0.9); // Cap at 90% of diff

                            z_new[i][j] = z[i][j] + u[face][i][j] * dt - erosion;
                        }
                    } else {
                        z_new[i][j] = z[i][j] + u[face][i][j] * dt;
                    }
                }
            }
            z = z_new;
            rn = computeRiverNetwork(z, face);
            A = computeDrainageArea(rn);
        }
    }

    // Generate heightmaps with uplift and erosion
    void generateHeightmaps() {
        generateInitialMaps();
        for (int face = 0; face < 6; ++face) {
            applyAnalyticalErosion(face);
        }
    }

    // Voxelization for Marching Cubes
    FaceST get_face_and_st(const Vec3& D) {
        double coords[3] = {D.x, D.y, D.z};  // Temporary array for subscript access
        double abs_D[3] = {std::abs(coords[0]), std::abs(coords[1]), std::abs(coords[2])};
        int k = 0;
        if (abs_D[1] > abs_D[k]) k = 1;
        if (abs_D[2] > abs_D[k]) k = 2;
        int sign = coords[k] > 0 ? 1 : -1;
        int face = k * 2 + (sign < 0 ? 1 : 0);
        int m = (k + 1) % 3;
        int n = (k + 2) % 3;
        double s = coords[m] / coords[k];
        double t = coords[n] / coords[k];
        return {face, s, t};
    }

    double interpolate(const Heightmap& hm, double u, double v) {
        int i0 = std::floor(u), j0 = std::floor(v);
        int i1 = i0 + 1, j1 = j0 + 1;
        if (i0 < 0) i0 = 0;
        if (j0 < 0) j0 = 0;
        if (i1 >= N) i1 = N - 1;
        if (j1 >= N) j1 = N - 1;
        double fu = u - i0, fv = v - j0;
        double h00 = hm[i0][j0], h10 = hm[i1][j0];
        double h01 = hm[i0][j1], h11 = hm[i1][j1];
        return (1 - fu) * (1 - fv) * h00 + fu * (1 - fv) * h10 +
               (1 - fu) * fv * h01 + fu * fv * h11;
    }

    void generateBaseTerrain() {
        int R_max = static_cast<int>(params.R * 1.1 + 0.5);
        openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
        for (int x = -R_max; x <= R_max; ++x) {
            for (int y = -R_max; y <= R_max; ++y) {
                for (int z = -R_max; z <= R_max; ++z) {
                    double r = std::sqrt(x * x + y * y + z * z);
                    if (r > 0 && r <= params.R * 1.1) {
                        Vec3 D = normalize(Vec3(x, y, z));
                        auto [face, s, t] = get_face_and_st(D);
                        double u = (s + 1.0) / 2.0 * (N - 1);
                        double v = (t + 1.0) / 2.0 * (N - 1);
                        double h = interpolate(heightmaps[face], u, v);
                        double s_value = r - (params.R + h);
                        accessor.setValue(openvdb::Coord(x, y, z), s_value);
                    }
                }
            }
        }
    }


};

class PlanetaryExporter {
public:
    // Export mesh to OBJ file
    static void exportMesh(const openvdb::FloatGrid::Ptr& grid, const std::string& filename) {
        openvdb::tools::VolumeToMesh mesher(0.0, 0.0);
        mesher(*grid);
        std::cout << "The cubes have marched " << std::endl;

        const auto& points = mesher.pointList();
        const auto& polygons = mesher.polygonPoolList();

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return;
        }

        // Write vertices
        for (int i = 0; i < mesher.pointListSize(); ++i) {
            const auto& p = points.get()[i];
            outfile << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
        }

        // Write faces: triangles and quads
        for (int i = 0; i < mesher.polygonPoolListSize(); ++i) {
            const auto& pool = polygons.get()[i];

            // Triangles
            for (size_t j = 0; j < pool.numTriangles(); ++j) {
                const auto& tri = pool.triangle(j);
                outfile << "f " << (tri[0] + 1) << " " << (tri[1] + 1) << " " << (tri[2] + 1) << "\n";
            }

            // Quads -> split into 2 triangles
            for (size_t j = 0; j < pool.numQuads(); ++j) {
                const auto& quad = pool.quad(j);
                // Triangle 1: 0-1-2
                outfile << "f " << (quad[0] + 1) << " " << (quad[1] + 1) << " " << (quad[2] + 1) << "\n";
                // Triangle 2: 0-2-3
                outfile << "f " << (quad[0] + 1) << " " << (quad[2] + 1) << " " << (quad[3] + 1) << "\n";
            }
        }

        outfile.close();
        std::cout << "Mesh exported successfully: " << filename << std::endl;
    }

    // Print sample values from a heightmap
    static void printHeightmapSample(const std::array<Heightmap, 6>& heightmaps, int N) {
        for (int facenum = 0; facenum < 6; ++facenum){
            const Heightmap& hm = heightmaps[facenum];  // Sample face 0 (+X)
            int center = N / 2;
            std::cout << "Sample 5x5 center values from heightmap (face " << facenum <<" ): \n";
            for (int i = center - 4; i <= center + 4; ++i) {
                for (int j = center - 4; j <= center + 4; ++j) {
                    std::cout << hm[i][j] << " ";
                }
                std::cout << "\n";
            }
            // Optional: Compute and print stats
            double min_h = *std::min_element(hm[facenum].begin(), hm[facenum].end());
            double max_h = *std::max_element(hm[facenum].begin(), hm[facenum].end());
            double avg_h = 0.0;
            for (const auto& row : hm) {
                for (double val : row) avg_h += val;
            }
            avg_h /= (N * N);
            std::cout << "Min height: " << min_h << ", Max height: " << max_h << ", Avg height: " << avg_h << "\n";
        }
    }

    // Save heightmap to CSV
    static void saveHeightmapToCSV(const std::array<Heightmap, 6>& heightmaps, int N) {
        for (int facenum = 0; facenum < 6; ++facenum) {
            std::string csv_filename = std::string("heightmap_face") + std::to_string(facenum) + std::string(".csv");
            const Heightmap &hm = heightmaps[facenum];  // Sample face 0
            std::ofstream outfile(csv_filename);
            if (!outfile.is_open()) {
                std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
                return;
            }
            outfile.imbue(std::locale("ru_RU.UTF-8"));  // Forces comma usage
            for (const auto &row: hm) {
                for (size_t j = 0; j < row.size(); ++j) {
                    outfile << row[j];
                    if (j < row.size() - 1) outfile << ";";
                }
                outfile << "\n";
            }
            outfile.close();
            std::cout << "Heightmap saved to " << csv_filename << std::endl;
        }
    }
};

int main() {
    setlocale(LC_NUMERIC, "French_Canada.1252");
    try {
        PlanetParams params = {9.81, 100.0, 100.0,  true, 4.5e9, 12345678, -30.0, 30.0};
        PlanetaryLandscape planet(params);
        planet.generateHeightmaps();
        
        // Use Exporter
        PlanetaryExporter::printHeightmapSample(planet.getHeightmaps(), planet.getResolution());
        PlanetaryExporter::saveHeightmapToCSV(planet.getHeightmaps(), planet.getResolution());
        
        planet.generateBaseTerrain();
        // planet.generateMesh(); // Removed redundancy
        
        PlanetaryExporter::exportMesh(planet.getGrid(), "planet2.obj");
    }
    catch (...) {
        std::cout << "something is wrong";
    }
    return 0;
}