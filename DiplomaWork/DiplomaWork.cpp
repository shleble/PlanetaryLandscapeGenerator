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
#include <chrono> // For timing

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
    std::array<Heightmap, 6> biomes;
    std::array<Heightmap, 6> z0;        // Initial heightmaps
    std::array<Heightmap, 6> u;         // Uplift maps
    std::array<Heightmap, 6> heightmaps; // Final eroded heightmaps
    FastNoiseLite noise;
    FastNoiseLite biomeNoise;
    FastNoiseLite caveNoise;
    std::mt19937 rng; // Seeded RNG

public:
    const openvdb::FloatGrid::Ptr& getGrid() const { return grid; }
    const std::array<Heightmap, 6>& getBiomes() const { return biomes; }
    const std::array<Heightmap, 6>& getHeightmaps() const { return heightmaps; }
    const PlanetParams& getParams() const { return params; }
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

        // Setup Biome Noise
        biomeNoise.SetSeed(params.S + 1); // Different seed
        biomeNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        biomeNoise.SetFrequency(1.0f / params.R);
        biomeNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
        biomeNoise.SetFractalOctaves(3);

        // Setup Cave Noise
        caveNoise.SetSeed(params.S + 2);
        caveNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        caveNoise.SetFrequency(8.0f / params.R);
        caveNoise.SetFractalType(FastNoiseLite::FractalType_Ridged);
        caveNoise.SetFractalOctaves(2);

        for (auto& hm : biomes) hm.resize(N, std::vector<double>(N, 0.0));
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

        for (int face = 0; face < 6; ++face) {
            for (int i = 0; i < N; ++i) {
                double s = -1.0 + 2.0 * i / (N - 1);
                for (int j = 0; j < N; ++j) {
                    double t = -1.0 + 2.0 * j / (N - 1);
                    Vec3 D = get_direction(face, s, t);
                    
                    // 1. Sample Biome Data
                    double biomeVal = biomeNoise.GetNoise(D.x * scale, D.y * scale, D.z * scale);
                    biomes[face][i][j] = biomeVal;
                    
                    // 2. Define Biome Modifiers
                    double z0_mult = 1.0;
                    double u_mult = 1.0;
                    double base_offset = 0.0;
                    
                    if (biomeVal < -0.2) { 
                        // OCEAN / BASIN
                        z0_mult = 0.3;
                        u_mult = 0.0;
                        base_offset = -params.R * 0.05;
                    } else if (biomeVal > 0.4) {
                        // MOUNTAINS
                        z0_mult = 1.2;
                        u_mult = 0.7;
                        base_offset = params.R * 0.08;
                    } else {
                        // PLAINS / HILLS
                        z0_mult = 0.8;
                        u_mult = 0.2;
                        base_offset = 0.0;
                    }

                    double noiseVal = noise.GetNoise(D.x * scale, D.y * scale, D.z * scale);

                    double baseline_amp = params.R * 0.15;

                    z0[face][i][j] = (noiseVal * baseline_amp * z0_mult) + base_offset;

                    double x = D.x * scale;
                    double y = D.y * scale;
                    double z = D.z * scale;
                    double u_noise = noise.GetNoise(x + 2000.0, y + 2000.0, z + 2000.0);
                    double baseline_uplift = params.R * 0.05;
                    u[face][i][j] = (u_noise * 0.5 + 0.5) * baseline_uplift * u_mult;

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
                    double u_dist = std::min(i, N - 1 - i) / (double)N;
                    double v_dist = std::min(j, N - 1 - j) / (double)N;
                    double edge_blend = std::min(1.0, std::min(u_dist, v_dist) * 8.0);

                    Cell r = rn[i][j];
                    if (r.i != -1) {
                        double slope = (z[i][j] - z[r.i][r.j]) / dx;
                        if (slope > 0) {
                            double erosion = k * std::pow(A[i][j], m) * std::pow(slope, n) * dt;
                            double max_erosion = z[i][j] - z[r.i][r.j];
                            erosion = std::min(erosion, max_erosion * 0.9);

                            z_new[i][j] = z[i][j] + (u[face][i][j] * dt - erosion) * edge_blend;
                        } else {
                            z_new[i][j] = z[i][j] + (u[face][i][j] * dt) * edge_blend;
                        }
                    } else {
                        z_new[i][j] = z[i][j] + (u[face][i][j] * dt) * edge_blend;
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
        
        std::cout << "Starting Analytical Erosion..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int face = 0; face < 6; ++face) {
            applyAnalyticalErosion(face);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Analysis Erosion completed in: " << elapsed.count() << " seconds." << std::endl;
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

    float estimateSDF(double x, double y, double z, double r) {
        Vec3 D = normalize(Vec3(x, y, z));
        auto [face, s, t] = get_face_and_st(D);
        double u_coord = (s + 1.0) / 2.0 * (N - 1);
        double v_coord = (t + 1.0) / 2.0 * (N - 1);
        double h = interpolate(heightmaps[face], u_coord, v_coord);
        return static_cast<float>(r - (params.R + h));
    }

    float computeSDF(double x, double y, double z, double r) {
        Vec3 D = normalize(Vec3(x, y, z));


        // 1. Domain Warping (Overhangs)
        double warpX = caveNoise.GetNoise((double)x * 0.5, (double)y * 0.5, (double)z);
        double warpY = caveNoise.GetNoise((double)y * 0.5, (double)z * 0.5, (double)x);
        double warpStrength = 15.0;

        Vec3 D_warped = normalize(Vec3(x + warpX * warpStrength, y + warpY * warpStrength, z));

        // 2. Sample Heightmap with Warped Coordinates
        auto [face, s, t] = get_face_and_st(D_warped);
        double u_coord = (s + 1.0) / 2.0 * (N - 1);
        double v_coord = (t + 1.0) / 2.0 * (N - 1);
        double h = interpolate(heightmaps[face], u_coord, v_coord);

        // Base SDF
        double s_value = r - (params.R + h);

        // 3. Terracing (Stratification)
        double terrace = std::sin((r - params.R) * 0.8) * 1.5;
        s_value += terrace;

        // 4. Arches (Surface Breaching Caves)
//        if (s_value > -5.0 && s_value < 10.0) {
//             double archVal = caveNoise.GetNoise((double)x * 1.2, (double)y * 1.2, (double)z * 1.2);
//             if (archVal > 0.65) {
//                 double hole_dist = (archVal - 0.65) * 20.0;
//                 s_value = std::max(s_value, hole_dist);
//             }
//        }


        return static_cast<float>(s_value);
    }

    // Full-grid generation
    void generateBaseTerrain() {
        int R_max = static_cast<int>(params.R * 1.5 + 0.5);
        openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
        for (int x = -R_max; x <= R_max; ++x) {
            for (int y = -R_max; y <= R_max; ++y) {
                for (int z = -R_max; z <= R_max; ++z) {
                    double r = std::sqrt((double)x * x + (double)y * y + (double)z * z);
                    if (r > 0 && r <= params.R * 1.5) {
                        float sdf = computeSDF(x, y, z, r);
                        if (std::abs(sdf) < 3.0f) {
                            accessor.setValue(openvdb::Coord(x, y, z), sdf);
                        }
                    }
                }
            }
        }
    }


};

class PlanetaryExporter {
public:
    static void exportMeshChunked(PlanetaryLandscape& planet, const std::string& filename, int chunk_size = 256) {
        const auto& params = planet.getParams();
        int R_max = static_cast<int>(params.R * 1.5 + 0.5);

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return;
        }

        std::string verts_file = filename + ".verts.tmp";
        std::string faces_file = filename + ".faces.tmp";
        std::ofstream verts_out(verts_file);
        std::ofstream faces_out(faces_file);

        if (!verts_out.is_open() || !faces_out.is_open()) {
            std::cerr << "Cannot create temp files for chunked export" << std::endl;
            return;
        }

        size_t vertex_offset = 0; // Running vertex index offset across chunks
        int total_chunks = 0;
        int x_min = -R_max;
        int x_max = R_max;

        float extract_band = 40.0f;
        float eval_band = 80.0f;

        for (int chunk_start = x_min; chunk_start <= x_max; chunk_start += chunk_size) {
            int chunk_end = std::min(chunk_start + chunk_size - 1, x_max);
            total_chunks++;

            std::cout << "  Chunk " << total_chunks << ": x=[" << chunk_start << ".." << chunk_end << "]" << std::flush;

            openvdb::FloatGrid::Ptr chunk_grid = openvdb::FloatGrid::create(extract_band);
            chunk_grid->setGridClass(openvdb::GRID_LEVEL_SET);
            openvdb::FloatGrid::Accessor accessor = chunk_grid->getAccessor();

            int pad = 3;
            for (int x = chunk_start - pad; x <= chunk_end + pad; ++x) {
                for (int y = -R_max; y <= R_max; ++y) {
                    for (int z = -R_max; z <= R_max; ++z) {
                        double r = std::sqrt((double)x * x + (double)y * y + (double)z * z);
                        if (r > 0 && r <= params.R * 1.5) {
                            float est_sdf = planet.estimateSDF(x, y, z, r);

                            if (std::abs(est_sdf) < eval_band) {
                                float sdf = planet.computeSDF(x, y, z, r);

                                if (std::abs(sdf) < extract_band) {
                                    accessor.setValue(openvdb::Coord(x, y, z), sdf);
                                } else if (sdf <= -extract_band) {
                                    accessor.setValueOff(openvdb::Coord(x, y, z), -extract_band);
                                }
                            }
                        }
                    }
                }
            }

            openvdb::tools::VolumeToMesh mesher(0.0, 0.0);
            mesher(*chunk_grid);

            size_t num_points = mesher.pointListSize();
            const auto& points = mesher.pointList();

            for (size_t i = 0; i < num_points; ++i) {
                const auto& p = points.get()[i];
                verts_out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
            }

            double x_lo = (double)chunk_start - 0.5;  // half-voxel tolerance
            double x_hi = (double)chunk_end + 0.5;

            const auto& polygons = mesher.polygonPoolList();
            for (int i = 0; i < mesher.polygonPoolListSize(); ++i) {
                const auto& pool = polygons.get()[i];

                for (size_t j = 0; j < pool.numTriangles(); ++j) {
                    const auto& tri = pool.triangle(j);
                    double cx = (points.get()[tri[0]][0] + points.get()[tri[1]][0] + points.get()[tri[2]][0]) / 3.0;

                    if (cx >= x_lo && cx < x_hi) {
                        faces_out << "f "
                                  << (tri[0] + 1 + vertex_offset) << " "
                                  << (tri[1] + 1 + vertex_offset) << " "
                                  << (tri[2] + 1 + vertex_offset) << "\n";
                    }
                }

                for (size_t j = 0; j < pool.numQuads(); ++j) {
                    const auto& quad = pool.quad(j);
                    double cx = (points.get()[quad[0]][0] + points.get()[quad[1]][0] +
                                 points.get()[quad[2]][0] + points.get()[quad[3]][0]) / 4.0;

                    if (cx >= x_lo && cx < x_hi) {
                        faces_out << "f "
                                  << (quad[0] + 1 + vertex_offset) << " "
                                  << (quad[1] + 1 + vertex_offset) << " "
                                  << (quad[2] + 1 + vertex_offset) << "\n";
                        faces_out << "f "
                                  << (quad[0] + 1 + vertex_offset) << " "
                                  << (quad[2] + 1 + vertex_offset) << " "
                                  << (quad[3] + 1 + vertex_offset) << "\n";
                    }
                }
            }

            vertex_offset += num_points;

            std::cout << " -> " << num_points << " verts (total: " << vertex_offset << ")" << std::endl;

        }

        verts_out.close();
        faces_out.close();

        std::cout << "Merging " << total_chunks << " chunks into " << filename << "..." << std::endl;

        {
            std::ifstream verts_in(verts_file);
            outfile << verts_in.rdbuf();
        }
        {
            std::ifstream faces_in(faces_file);
            outfile << faces_in.rdbuf();
        }

        outfile.close();

        std::remove(verts_file.c_str());
        std::remove(faces_file.c_str());

        std::cout << "Mesh exported successfully: " << filename
                  << " (" << vertex_offset << " total vertices, " << total_chunks << " chunks)" << std::endl;
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
        auto total_start = std::chrono::high_resolution_clock::now();

        PlanetParams params = {9.81, 2000.0, 100.0,  true, 4.5e9, 12345678, -30.0, 30.0};
        PlanetaryLandscape planet(params);
        planet.generateHeightmaps();

        // Use Exporter
        PlanetaryExporter::printHeightmapSample(planet.getBiomes(), planet.getResolution());
        PlanetaryExporter::saveHeightmapToCSV(planet.getBiomes(), planet.getResolution());

        // Chunked SDF generation + meshing (RAM-safe for large planets)
        std::cout << "Exporting Mesh (chunked)..." << std::endl;
        auto export_start = std::chrono::high_resolution_clock::now();
        PlanetaryExporter::exportMeshChunked(planet, "planet2.obj");
        auto export_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> export_elapsed = export_end - export_start;
        std::cout << "Mesh Generation and Export completed in: " << export_elapsed.count() << " seconds." << std::endl;

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = total_end - total_start;
        std::cout << "TOTAL Execution Time: " << total_elapsed.count() << " seconds." << std::endl;
    }
    catch (...) {
        std::cout << "something is wrong";
    }
    return 0;
}