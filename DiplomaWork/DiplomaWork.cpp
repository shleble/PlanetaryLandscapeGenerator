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
#include <future>

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
    std::array<Heightmap, 6> biomes;    // Initial heightmaps for biome split
    std::array<Heightmap, 6> z0;        // Initial heightmaps
    std::array<Heightmap, 6> u;         // Uplift maps
    std::array<Heightmap, 6> heightmaps; // Final eroded heightmaps
    FastNoiseLite noise;
    FastNoiseLite biomeNoise; // Separate noise for macro-biomes
    FastNoiseLite caveNoise;  // 3D Noise for caves
    FastNoiseLite archNoise;  // 3D Noise for arches
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

        // Setup Biome Noise (Low Frequency, Macro structures)
        biomeNoise.SetSeed(params.S + 1); // Different seed
        biomeNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        biomeNoise.SetFrequency(1.0f / params.R);
        biomeNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
        biomeNoise.SetFractalOctaves(3);

        // Setup Cave Noise (3D Volumetric)
        caveNoise.SetSeed(params.S + 2);
        caveNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        caveNoise.SetFrequency(8.0f / params.R);
        caveNoise.SetFractalType(FastNoiseLite::FractalType_Ridged);
        caveNoise.SetFractalOctaves(2);

        archNoise.SetSeed(params.S + 3);
        archNoise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        archNoise.SetFrequency(4.0f / params.R);
        archNoise.SetFractalType(FastNoiseLite::FractalType_FBm);
        archNoise.SetFractalOctaves(2);

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
                        z0_mult = 0.3;     // Smooth bottom
                        u_mult = 0.0;      // No uplift
                        base_offset = -params.R * 0.05; // Lower ground
                    } else if (biomeVal > 0.4) {
                        // MOUNTAINS
                        z0_mult = 1.2;     // Rugged
                        u_mult = 0.7;      // High uplift
                        base_offset = params.R * 0.08; // Higher ground
                    } else {
                        // PLAINS / HILLS
                        z0_mult = 0.8;
                        u_mult = 0.2;
                        base_offset = 0.0;
                    }

                    double noiseVal = noise.GetNoise(D.x * scale, D.y * scale, D.z * scale);

                    // 3. Apply Modifiers
                    double baseline_amp = params.R * 0.15;

                    z0[face][i][j] = (noiseVal * baseline_amp * z0_mult) + base_offset;

                    // Uplift
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

    // Planchon-Darboux depression filling
    Heightmap fillDepressions(const Heightmap& z) const {
        Heightmap W(N, std::vector<double>(N, 1e9));
        using Element = std::tuple<double, int, int>;
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                    W[i][j] = z[i][j];
                    pq.push({W[i][j], i, j});
                }
            }
        }

        double epsilon = 1e-5;

        while (!pq.empty()) {
            auto [w, i, j] = pq.top();
            pq.pop();

            if (w > W[i][j]) continue;

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                        double new_w = std::max(z[ni][nj], W[i][j] + epsilon);
                        if (new_w < W[ni][nj]) {
                            W[ni][nj] = new_w;
                            pq.push({new_w, ni, nj});
                        }
                    }
                }
            }
        }
        return W;
    }

    // Compute river network for a face
    std::vector<std::vector<Cell>> computeRiverNetwork(const Heightmap& z, int face, std::mt19937& local_rng) {
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
                    receivers[i][j] = candidates[dist(local_rng)];
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
        std::mt19937 local_rng(params.S + face);
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
        Heightmap W = fillDepressions(z);
        auto rn = computeRiverNetwork(W, face, local_rng);
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
                            double max_erosion = z[i][j] - z[r.i][r.j];
                            erosion = std::min(erosion, max_erosion * 0.9);

                            z_new[i][j] = z[i][j] + (u[face][i][j] * dt - erosion);
                        } else {
                            z_new[i][j] = z[i][j] + (u[face][i][j] * dt);
                        }
                    } else {
                        z_new[i][j] = z[i][j] + (u[face][i][j] * dt);
                    }
                }
            }
            z = z_new;
            Heightmap W_new = fillDepressions(z);
            rn = computeRiverNetwork(W_new, face, local_rng);
            A = computeDrainageArea(rn);
        }
    }

    // Generate heightmaps with uplift and erosion
    void generateHeightmaps() {
        generateInitialMaps();

        std::cout << "Starting Analytical Erosion (Multithreaded)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::future<void>> futures;
        for (int face = 0; face < 6; ++face) {
            futures.push_back(std::async(std::launch::async, [this, face]() {
                this->applyAnalyticalErosion(face);
            }));
        }
        
        for (auto& f : futures) {
            f.get();
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

    double interpolate(const Heightmap& hm, double u, double v) const {
        int i0 = std::floor(u), j0 = std::floor(v);
        int i1 = i0 + 1, j1 = j0 + 1;
        double fu = u - i0, fv = v - j0;
        i0 = std::max(0, std::min(N - 1, i0));
        j0 = std::max(0, std::min(N - 1, j0));
        i1 = std::max(0, std::min(N - 1, i1));
        j1 = std::max(0, std::min(N - 1, j1));

        double h00 = hm[i0][j0], h10 = hm[i1][j0];
        double h01 = hm[i0][j1], h11 = hm[i1][j1];
        return (1 - fu) * (1 - fv) * h00 + fu * (1 - fv) * h10 +
               (1 - fu) * fv * h01 + fu * fv * h11;
    }

    // Blend physics heights across adjacent cubemap faces
    double get_blended_height(const Vec3& D) const {
        double abs_D[3] = {std::abs(D.x), std::abs(D.y), std::abs(D.z)};
        double coords[3] = {D.x, D.y, D.z};

        double blend_power = 6.0;
        double w[3] = {
                std::pow(abs_D[0], blend_power),
                std::pow(abs_D[1], blend_power),
                std::pow(abs_D[2], blend_power)
        };
        double total_w = w[0] + w[1] + w[2];
        w[0] /= total_w;
        w[1] /= total_w;
        w[2] /= total_w;

        double h = 0.0;
        for (int k = 0; k < 3; ++k) {
            if (w[k] > 0.001) {
                int sign = coords[k] > 0 ? 1 : -1;
                int face = k * 2 + (sign < 0 ? 1 : 0);
                int m = (k + 1) % 3;
                int n = (k + 2) % 3;

                double s = (coords[k] != 0.0) ? (coords[m] / coords[k]) : 0.0;
                double t = (coords[k] != 0.0) ? (coords[n] / coords[k]) : 0.0;

                double u_coord = (s + 1.0) / 2.0 * (N - 1);
                double v_coord = (t + 1.0) / 2.0 * (N - 1);
                h += w[k] * interpolate(heightmaps[face], u_coord, v_coord);
            }
        }
        return h;
    }


    // Fast base SDF estimation without 3D noise
    float estimateSDF(double x, double y, double z, double r) const {
        Vec3 D = normalize(Vec3(x, y, z));
        double h = get_blended_height(D);
        return static_cast<float>(r - (params.R + h));
    }

    // Compute the SDF value for a single voxel at (x, y, z) with precomputed radius r
    float computeSDF(double x, double y, double z, double r) const {
        Vec3 D = normalize(Vec3(x, y, z));

        // 1. Evaluate Pure Topography
        double h = get_blended_height(D);
        double s_value = r - (params.R + h);

        double feature_scale = params.R / 4000.0;

        // 2. Terracing
        double terrace = std::sin((r - params.R) * (0.8 / feature_scale)) * (1.5 * feature_scale);
        s_value += terrace;

        // 3. Spaced-out Volumetric Overhangs & Outcroppings
        double a_noise = archNoise.GetNoise((double)x, (double)y, (double)z);
        
        if (a_noise > 0.3) {
            double outcropping_intensity = (a_noise - 0.3) * 2.0; 
            double distance_from_surface = std::abs(s_value);
            double density_mask = std::clamp(1.0 - (distance_from_surface / (50.0 * feature_scale)), 0.0, 1.0);

            double rock_shape = std::abs(caveNoise.GetNoise((double)x * 1.5, (double)y * 1.5, (double)z * 1.5));
            s_value -= rock_shape * outcropping_intensity * (30.0 * feature_scale) * density_mask;
        }

        // 4. Sweeping Caverns & Arches (Swiss Cheese Boolean Subtraction)
        if (s_value > (-20.0 * feature_scale) && s_value < (40.0 * feature_scale) && a_noise > 0.5) {
            
            double void_dist = (a_noise - 0.5) * (120.0 * feature_scale);
            if (s_value < 0.0) {
                 double fade = 1.0 - (s_value / (-20.0 * feature_scale));
                 void_dist *= std::clamp(fade, 0.0, 1.0);
            }
            
            s_value = std::max(s_value, void_dist);
        }

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
    // Export mesh to OBJ file using chunked processing to limit RAM usage
    static void exportMeshChunked(PlanetaryLandscape& planet, const std::string& filename, int chunk_size = 1024, double voxel_size = -1.0) {
        const auto& params = planet.getParams();

        // Auto-scale sparsity
        if (voxel_size <= 0.0) {
            voxel_size = std::max(1.0, params.R / 500.0);
        }

        int R_max = static_cast<int>(params.R * 1.5 + 0.5);
        double phys_chunk_size = chunk_size * voxel_size;

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return;
        }
        // Collect all vertices first and then faces
        // Write vertices to a temp file, faces to another temp file, then concatenate

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
        double x_min = -R_max;
        double x_max = R_max;

        // Band parameters scaled securely relative to voxel_size
        float extract_band = 10.0f + 25.0f * voxel_size;
        float eval_band = 100.0f + 50.0f * voxel_size;

        for (double chunk_start = x_min; chunk_start <= x_max; chunk_start += phys_chunk_size) {
            double chunk_end = std::min(chunk_start + phys_chunk_size, x_max);
            total_chunks++;

            std::cout << "  Chunk " << total_chunks << ": x=[" << chunk_start << ".." << chunk_end << "]" << std::flush;

            // Build a temporary grid for this slab
            openvdb::FloatGrid::Ptr chunk_grid = openvdb::FloatGrid::create(extract_band);
            chunk_grid->setGridClass(openvdb::GRID_LEVEL_SET);
            // Apply scale transform so exporting directly outputs into Physical world coordinates
            chunk_grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
            // Populate SDF for this X-slab with padding
            int pad_idx = 5;
            int start_idx = std::floor(chunk_start / voxel_size);
            int end_idx = std::ceil(chunk_end / voxel_size);
            int yz_max_idx = std::ceil(R_max / voxel_size);

            int i_min = start_idx - pad_idx;
            int i_max = end_idx + pad_idx;
            int total_i = i_max - i_min + 1;

            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;

            std::vector<std::future<openvdb::FloatGrid::Ptr>> grid_futures;
            int block_size = (total_i + num_threads - 1) / num_threads;

            for (unsigned int t = 0; t < num_threads; ++t) {
                int thread_i_start = i_min + t * block_size;
                int thread_i_end = std::min(i_max, thread_i_start + block_size - 1);
                
                if (thread_i_start > i_max) break;

                grid_futures.push_back(std::async(std::launch::async, [thread_i_start, thread_i_end, yz_max_idx, voxel_size, &planet, &params, eval_band, extract_band]() {
                    openvdb::FloatGrid::Ptr local_grid = openvdb::FloatGrid::create(extract_band);
                    local_grid->setGridClass(openvdb::GRID_LEVEL_SET);
                    local_grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
                    openvdb::FloatGrid::Accessor local_accessor = local_grid->getAccessor();

                    for (int i = thread_i_start; i <= thread_i_end; ++i) {
                        for (int j = -yz_max_idx; j <= yz_max_idx; ++j) {
                            for (int k = -yz_max_idx; k <= yz_max_idx; ++k) {
                                double x = i * voxel_size;
                                double y = j * voxel_size;
                                double z = k * voxel_size;

                                double r = std::sqrt(x * x + y * y + z * z);
                                if (r > 0 && r <= params.R * 1.5) {
                                    float est_sdf = planet.estimateSDF(x, y, z, r);

                                    if (std::abs(est_sdf) < eval_band) {
                                        float sdf = planet.computeSDF(x, y, z, r);

                                        if (std::abs(sdf) < extract_band) {
                                            local_accessor.setValue(openvdb::Coord(i, j, k), sdf);
                                        } else if (sdf <= -extract_band) {
                                            local_accessor.setValueOff(openvdb::Coord(i, j, k), -extract_band);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return local_grid;
                }));
            }

            // Merge all thread-local grids into the main chunk grid
            for (auto& fut : grid_futures) {
                openvdb::FloatGrid::Ptr local_grid = fut.get();
                chunk_grid->tree().merge(local_grid->tree());
            }

            // Mesh this chunk
            openvdb::tools::VolumeToMesh mesher(0.0, 0.0);
            mesher(*chunk_grid);

            size_t num_points = mesher.pointListSize();
            const auto& points = mesher.pointList();

            // Write ALL vertices from this chunk
            for (size_t i = 0; i < num_points; ++i) {
                const auto& p = points.get()[i];
                verts_out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
            }

            // Write faces, but ONLY those whose centroid X falls within
            double x_lo = chunk_start;
            double x_hi = chunk_end;

            const auto& polygons = mesher.polygonPoolList();
            for (int i = 0; i < mesher.polygonPoolListSize(); ++i) {
                const auto& pool = polygons.get()[i];

                for (size_t j = 0; j < pool.numTriangles(); ++j) {
                    const auto& tri = pool.triangle(j);
                    // Compute centroid X of this triangle
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

            // chunk_grid goes out of scope here and is freed
        }

        verts_out.close();
        faces_out.close();

        // Concatenate: vertices first, then faces
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

        // Clean up temp files
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

        PlanetParams params = {9.81, 300.0, 100.0,  true, 4.5e9, 12345678, -30.0, 30.0};
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