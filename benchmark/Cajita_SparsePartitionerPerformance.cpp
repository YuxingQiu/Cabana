/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "Cabana_BenchmarkUtils.hpp"

#include <Cajita_SparseDimPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Helper functions.
#define PARTICLE_WORKLOAD 0
#define SPARSE_MAP_WOPRKLOAD 1

std::set<std::array<int, 3>> generateRandomTiles( int tiles_per_dim,
                                                  int tile_num )
{
    std::set<std::array<int, 3>> tiles_set;
    while ( static_cast<int>( tiles_set.size() ) < tile_num )
    {
        int rand_tile[3];
        for ( int d = 0; d < 3; ++d )
            rand_tile[d] = std::rand() % tiles_per_dim;
        tiles_set.insert( rand_tile );
    }
    return tiles_set;
}

int current = 0;
int uniqueNumber() { return current++; }

std::set<std::array<int, 3>> generateRandomTileSequence( int tiles_per_dim )
{
    std::set<std::array<int, 3>> tiles_set;
    std::vector<int> random_seq( num_tiles_per_dim );

    std::generate_n( random_seq.data(), tiles_per_dim, uniqueNumber );
    for ( int d = 0; d < 3; ++d )
    {
        std::random_shuffle( random_seq.begin(), random_seq.end() );
        for ( int n = 0; n < tiles_per_dim; ++n )
        {
            // pass
        }
    }

    while ( static_cast<int>( tiles_set.size() ) < tile_num )
    {
        int rand_tile[3];
        for ( int d = 0; d < 3; ++d )
            rand_tile[d] = std::rand() % tiles_per_dim;
        tiles_set.insert( rand_tile );
    }
    return tiles_set;
}

//---------------------------------------------------------------------------//
// Performance test.
template <class Device, unsigned WLType,
          typename std::enable_if_t<WLType == PARTICLE_WORKLOAD> *= nullptr>
void performanceTest( std::ostream& stream, const std::string& test_prefix )
{
    // pass
}

template <class Device, unsigned WLType,
          typename std::enable_if_t<WLType == SPARSE_MAP_WOPRKLOAD> *= nullptr>
void performanceTest( std::ostream& stream, const std::string& test_prefix )
{
    // Domain size setup
    std::array<float, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<float, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    constexpr int cell_num_per_tile_dim = 4;
    constexpr int cell_bits_per_tile_dim = 2;

    // Declare the fraction of occupied tiles in the whole domain
    std::vector<double> occupy_fraction = { 0.001, 0.005, 0.01, 0.05, 0.1,
                                            0.25,  0.5,   0.75, 1.0 };
    int occupy_fraction_size = occupy_fraction.size();

    // Declare the size (cell nums) of the domain
    std::vector<int> num_cells_per_dim = { 16, 32, 64, 128, 256, 512, 1024 };
    int num_cells_per_dim_size = num_cells_per_dim.size();

    // Number of runs in the test loops.
    int num_run = 10;

    for ( int c = 0; c < num_cells_per_dim_size; ++c )
    {
        // init the sparse grid domain
        std::array<int, 3> global_num_cell = {
            num_cells_per_dim[c], num_cells_per_dim[c], num_cells_per_dim[c] };
        auto global_mesh = createSparseGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        float cell_size = 1.0 / num_cells_per_dim[c];
        int num_tiles_per_dim = num_cells_per_dim[c] >> cell_bits_per_tile_dim;

        // create sparse map
        int pre_alloc_size = num_cells_per_dim[c] * num_cells_per_dim[c];
        auto sis = createSparseMap<typename Device::execution_space>(
            global_mesh, pre_alloc_size );

        // Generate a random set of occupied tiles
        auto tiles_set = generateRandomTileSequence( num_cells_per_dim[c] );
    }
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument - file name for output \n \
             \n \
             Example: \n \
             $/: ./SparseMapPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Barier before continuing.
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Get Cartesian comm
    std::array<int, 3> ranks_per_dim;
    for ( std::size_t d = 0; d < 3; ++d )
        ranks_per_dim[d] = 0;
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Open the output file on rank 0.
    std::fstream file;
    if ( 0 == comm_rank )
        file.open( filename, std::fstream::out );

    // Output problem details.
    if ( 0 == comm_rank )
    {
        file << "\n";
        file << "Cajita Sparse Partitioner Performance Benchmark"
             << "\n";
        file << "----------------------------------------------"
             << "\n";
        file << "MPI Ranks: " << comm_size << "\n";
        file << "MPI Cartesian Dim Ranks: (" << ranks_per_dim[0] << ", "
             << ranks_per_dim[1] << ", " << ranks_per_dim[2] << ")\n";
        file << "----------------------------------------------"
             << "\n";
        file << "\n";
        file << std::flush;
    }

    // Run the tests.
#ifdef KOKKOS_ENABLE_SERIAL
    using SerialDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    performanceTest<SerialDevice, PARTICLE_WORKLOAD>( file, "serial_parWL_" );
    performanceTest<SerialDevice, SPARSE_MAP_WOPRKLOAD>( file,
                                                         "serial_smapWL_" );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice, PARTICLE_WORKLOAD>( file, "openmp_parWL_" );
    performanceTest<OpenMPDevice, SPARSE_MAP_WOPRKLOAD>( file,
                                                         "openmp_smapWL_" );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    // using CudaUVMDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>;
    performanceTest<CudaDevice, PARTICLE_WORKLOAD>( file, "cuda_parWL_" );
    performanceTest<CudaDevice, SPARSE_MAP_WOPRKLOAD>( file, "cuda_smapWL_" );
    // performanceTest<CudaDevice>( file, "cudauvm_" );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}