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

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream, const std::string& test_prefix )
{
    // pass
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
    performanceTest<SerialDevice>( file, "serial_" );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( file, "openmp_" );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    // using CudaUVMDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>;
    performanceTest<CudaDevice>( file, "cuda_" );
    // performanceTest<CudaDevice>( file, "cudauvm_" );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}