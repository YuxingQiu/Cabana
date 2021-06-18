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
struct ParticleWorkloadTag
{
};
struct SparseMapTag
{
};

std::set<std::array<int, 3>> generateRandomTiles( int tiles_per_dim,
                                                  int tile_num )
{
    std::set<std::array<int, 3>> tiles_set;
    while ( static_cast<int>( tiles_set.size() ) < tile_num )
    {
        std::array<int, 3> rand_tile;
        for ( int d = 0; d < 3; ++d )
            rand_tile[d] = std::rand() % tiles_per_dim;
        tiles_set.insert( rand_tile );
    }
    return tiles_set;
}

// generate a random tile sequence
int current = 0;
int uniqueNumber() { return current++; }

Kokkos::View<int* [3], Kokkos::HostSpace>
generateRandomTileSequence( int tiles_per_dim ) {
    Kokkos::View<int* [3], Kokkos::HostSpace> tiles_host(
        "random_tile_sequence_host",
        tiles_per_dim * tiles_per_dim * tiles_per_dim );

    std::vector<int> random_seq( tiles_per_dim );
    std::generate_n( random_seq.data(), tiles_per_dim, uniqueNumber );
    for ( int d = 0; d < 3; ++d )
    {
        std::shuffle( random_seq.begin(), random_seq.end(),
                      std::default_random_engine( 3439203991 ) );
        for ( int n = 0; n < tiles_per_dim; ++n )
            tiles_host( n, d ) = random_seq[n];
    }
    return tiles_host;
}

// generate average partitioner
std::array<std::vector<int>, 3> computeAveragePartition(
    const int tile_per_dim, const std::array<int, 3>& ranks_per_dim )
{
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( tile_per_dim );
    }
    return rec_partitions;
}

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( ParticleWorkloadTag, std::ostream& stream, MPI_Comm comm,
                      const std::string& test_prefix )
{
    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Domain size setup
    std::array<float, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<float, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    constexpr int cell_num_per_tile_dim = 4;
    constexpr int cell_bits_per_tile_dim = 2;

    // Declare the totla number of particles
    std::vector<int> num_particles = { 100,    1000,   10000,
                                       100000, 500000, 1000000 };
    int num_particles_size = num_particles.size();

    // Declare the size (cell nums) of the domain
    std::vector<int> num_cells_per_dim = { 16, 32, 64, 128, 256, 512, 1024 };
    int num_cells_per_dim_size = num_cells_per_dim.size();

    // Number of runs in the test loops.
    int num_run = 10;

    // Basic settings for partitioenr
    float max_workload_coeff = 1.5;
    int max_optimize_iteration = 10;
    int num_step_rebalance = 100;

    // compute the max number of particles handled by the current MPI rank
    int max_par_num = num_particles.back() / comm_size +
                      ( num_particles.back() % comm_size < comm_rank ? 1 : 0 );

    // Generate a random set of particles
    auto poses_host = Cabana::Benchmark::createParticles<float>(
        max_par_num, global_low_corner, global_high_corner );
    auto poses = Kokkos::create_mirror_view_and_copy(
        typename Device::memory_space(), poses_host );

    for ( int c = 0; c < num_cells_per_dim_size; ++c )
    {
        // init the sparse grid domain
        std::array<int, 3> global_num_cell = {
            num_cells_per_dim[c], num_cells_per_dim[c], num_cells_per_dim[c] };
        int num_tiles_per_dim = num_cells_per_dim[c] >> cell_bits_per_tile_dim;

        // set up partitioner
        Cajita::SparseDimPartitioner<Device, cell_num_per_tile_dim> partitioner(
            comm, max_workload_coeff, max_par_num, num_step_rebalance,
            global_num_cell, max_optimize_iteration );
        auto ranks_per_dim =
            partitioner.ranksPerDimension( comm, global_num_cell );
        auto ave_partition =
            computeAveragePartition( num_tiles_per_dim, ranks_per_dim );

        // Create insertion timers
        std::stringstream local_workload_name;
        local_workload_name << test_prefix << "compute_local_workload_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer local_workload_timer(
            local_workload_name.str(), num_particles_size );

        std::stringstream prefix_sum_name;
        prefix_sum_name << test_prefix << "compute_prefix_sum_"
                        << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer prefix_sum_timer( prefix_sum_name.str(),
                                                   num_particles_size );

        std::stringstream total_optimize_name;
        total_optimize_name << test_prefix << "total_optimize_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer total_optimize_timer(
            total_optimize_name.str(), num_particles_size );

        std::stringstream single_optimize_name;
        single_optimize_name << test_prefix << "single_step_optimize_"
                             << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer single_optimize_timer(
            single_optimize_name.str(), num_particles_size );

        // loop over all the particle numbers
        for ( int p = 0; p < num_particles_size; ++p )
        {
            // compute the number of particles handled by the current MPI rank
            int par_num = num_particles[p] / comm_size +
                          ( num_particles[p] % comm_size < comm_rank ? 1 : 0 );

            auto pos_view =
                Kokkos::subview( poses, Kokkos::pair<int, int>( 0, par_num ),
                                 Kokkos::pair<int, int>( 0, 3 ) );

            // try for num_run times
            for ( int t = 0; t < num_run; ++t )
            {
                // ensure every optimization process starts from the same status
                partitioner.initializeRecPartition(
                    ave_partition[0], ave_partition[1], ave_partition[2] );

                // compute local workload
                local_workload_timer.start( p );
                partitioner.computeLocalWorkLoad( pos_view, par_num,
                                                  global_low_corner,
                                                  1.0f / num_cells_per_dim[c] );
                local_workload_timer.stop( p );

                // compute prefix sum matrix
                prefix_sum_timer.start( p );
                partitioner.computeFullPrefixSum( comm );
                prefix_sum_timer.stop( p );

                // optimization
                bool is_changed = false;
                // full timer
                total_optimize_timer.start( p );
                for ( int i = 0; i < max_optimize_iteration; ++i )
                {
                    // timer for a single optimizing step
                    single_optimize_timer.start( p );
                    partitioner.optimizePartition( is_changed,
                                                   std::rand() % 3 );
                    single_optimize_timer.stop( p );
                    if ( !is_changed )
                        break;
                }
                total_optimize_timer.stop( p );
            }
        }
        // Output results
        outputResults( stream, "insert_tile_num", num_particles,
                       local_workload_timer, comm );
        outputResults( stream, "insert_tile_num", num_particles,
                       prefix_sum_timer, comm );
        outputResults( stream, "insert_tile_num", num_particles,
                       single_optimize_timer, comm );
        outputResults( stream, "insert_tile_num", num_particles,
                       total_optimize_timer, comm );
        stream << std::flush;
    }
}

template <class Device>
void performanceTest( SparseMapTag, std::ostream& stream, MPI_Comm comm,
                      const std::string& test_prefix )
{
    using exec_space = typename Device::execution_space;
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

    // Basic settings for partitioenr
    float max_workload_coeff = 1.5;
    int max_optimize_iteration = 10;
    int num_step_rebalance = 100;

    for ( int c = 0; c < num_cells_per_dim_size; ++c )
    {
        // init the sparse grid domain
        std::array<int, 3> global_num_cell = {
            num_cells_per_dim[c], num_cells_per_dim[c], num_cells_per_dim[c] };
        auto global_mesh = Cajita::createSparseGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        float cell_size = 1.0 / num_cells_per_dim[c];
        int num_tiles_per_dim = num_cells_per_dim[c] >> cell_bits_per_tile_dim;

        // create sparse map
        int pre_alloc_size = num_cells_per_dim[c] * num_cells_per_dim[c];
        auto sis =
            Cajita::createSparseMap<exec_space>( global_mesh, pre_alloc_size );

        // Generate a random set of occupied tiles
        auto tiles_host = generateRandomTileSequence( num_tiles_per_dim );
        auto tiles_view = Kokkos::create_mirror_view_and_copy(
            typename Device::memory_space(), tiles_host );

        // set up partitioner
        auto total_num =
            num_tiles_per_dim * num_tiles_per_dim * num_tiles_per_dim;
        Cajita::SparseDimPartitioner<Device, cell_num_per_tile_dim> partitioner(
            comm, max_workload_coeff, total_num, num_step_rebalance,
            global_num_cell, max_optimize_iteration );
        auto ranks_per_dim =
            partitioner.ranksPerDimension( comm, global_num_cell );
        auto ave_partition =
            computeAveragePartition( num_tiles_per_dim, ranks_per_dim );

        // Create insertion timers
        std::stringstream local_workload_name;
        local_workload_name << test_prefix << "compute_local_workload_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer local_workload_timer(
            local_workload_name.str(), occupy_fraction_size );

        std::stringstream prefix_sum_name;
        prefix_sum_name << test_prefix << "compute_prefix_sum_"
                        << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer prefix_sum_timer( prefix_sum_name.str(),
                                                   occupy_fraction_size );

        std::stringstream total_optimize_name;
        total_optimize_name << test_prefix << "total_optimize_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer total_optimize_timer(
            total_optimize_name.str(), occupy_fraction_size );

        std::stringstream single_optimize_name;
        single_optimize_name << test_prefix << "single_step_optimize_"
                             << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer single_optimize_timer(
            single_optimize_name.str(), occupy_fraction_size );

        // loop over all the occupy_fractions
        for ( int frac = 0; frac < occupy_fraction_size; ++frac )
        {
            // compute the number of distinct tiles that will be inserted to the
            // sparse map
            int num_insert =
                static_cast<int>( occupy_fraction[frac] * num_tiles_per_dim *
                                  num_tiles_per_dim * num_tiles_per_dim );

            // register selected tiles to the sparseMap
            Kokkos::parallel_for(
                "insert_tile_to_sparse_map",
                Kokkos::RangePolicy<exec_space>( 0, num_insert ),
                KOKKOS_LAMBDA( const int id ) {
                    sis.insertTile( tiles_view( id, 0 ), tiles_view( id, 1 ),
                                    tiles_view( id, 2 ) );
                } );

            for ( int t = 0; t < num_run; ++t )
            {
                // ensure every optimization process starts from the same status
                partitioner.initializeRecPartition(
                    ave_partition[0], ave_partition[1], ave_partition[2] );

                // compute local workload
                local_workload_timer.start( frac );
                partitioner.computeLocalWorkLoad( sis );
                local_workload_timer.stop( frac );

                // compute prefix sum matrix
                prefix_sum_timer.start( frac );
                partitioner.computeFullPrefixSum( comm );
                prefix_sum_timer.stop( frac );

                // optimization
                bool is_changed = false;
                // full timer
                total_optimize_timer.start( frac );
                for ( int i = 0; i < max_optimize_iteration; ++i )
                {
                    // timer for a single optimizing step
                    single_optimize_timer.start( frac );
                    partitioner.optimizePartition( is_changed,
                                                   std::rand() % 3 );
                    single_optimize_timer.stop( frac );
                    if ( !is_changed )
                        break;
                }
                total_optimize_timer.stop( frac );
            }
        }

        // Output results
        outputResults( stream, "insert_tile_num", occupy_fraction,
                       local_workload_timer, comm );
        outputResults( stream, "insert_tile_num", occupy_fraction,
                       prefix_sum_timer, comm );
        outputResults( stream, "insert_tile_num", occupy_fraction,
                       single_optimize_timer, comm );
        outputResults( stream, "insert_tile_num", occupy_fraction,
                       total_optimize_timer, comm );
        stream << std::flush;
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
    performanceTest<SerialDevice>( ParticleWorkloadTag(), file, MPI_COMM_WORLD,
                                   "serial_parWL_" );
    performanceTest<SerialDevice>( SparseMapTag(), file, MPI_COMM_WORLD,
                                   "serial_smapWL_" );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( ParticleWorkloadTag(), file, MPI_COMM_WORLD,
                                   "openmp_parWL_" );
    performanceTest<OpenMPDevice>( SparseMapTag(), file, MPI_COMM_WORLD,
                                   "openmp_smapWL_" );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    performanceTest<CudaDevice>( ParticleWorkloadTag(), file, MPI_COMM_WORLD,
                                 "cuda_parWL_" );
    performanceTest<CudaDevice>( SparseMapTag(), file, MPI_COMM_WORLD,
                                 "cuda_smapWL_" );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}