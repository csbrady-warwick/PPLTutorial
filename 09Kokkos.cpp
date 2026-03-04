#include <iostream>
#include <vector>
#include <Kokkos_Core.hpp>
#include "timer.h"

//A kokkos execution space is a type that represents where the code should run
#ifdef KOKKOS_CUDA
//Run on an NVIDIA GPU if KOKKOS_CUDA is defined
    using executionSpace = Kokkos::Cuda;
#else
//Run on the CPU with OpenMP if KOKKOS_CUDA is not defined
    using executionSpace = Kokkos::OpenMP;
#endif

//There are other execution spaces available in Kokkos, for example there is a HIP backend for AMD GPUs and a SYCL backend that can run on a variety of devices. You can also write your own execution space if you have some custom hardware that you want to run on

//Once you have an execution space defined there is also a memory space that goes along with it. This is the type of memory that the data will be allocated in. For example, if you are running on a GPU then the memory space will be device memory, and if you are running on the CPU then the memory space will be host memory.
using memorySpace = executionSpace::memory_space;
using retreivedExecutionSpace = memorySpace::execution_space;


// Iterate over the array and apply the functor to each element
// Use OpenMP to parallelize the outer loop
template <typename T>
void iterate(T functor, int lbound1, int lbound2, int ubound1, int ubound2)
{
    using iterPolicy = Kokkos::MDRangePolicy<executionSpace, Kokkos::Rank<2>>;
    Kokkos::parallel_for("iterate", iterPolicy({lbound1, lbound2}, {ubound1, ubound2}), functor);
}

#define NX 4000
#define NY 4000
#define NITS 300

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        //Kokkos::view is a Kokkos version of mdspan that automatically works on both CPU and GPU. Unlike the mdspan it does manage memory, automatically deallocating it when it goes out of scope. That means that we have to make sure that it goes out of scope before we call Kokkos::finalize() at the end of main
        //That is what the extra set of curly braces is doing here, it creates a scope for the views to be destroyed before we finalize Kokkos
        //You pass the memory space as a template parameter which tells Kokkos where to allocate the memory backing the view. The "double**" doesn't really have anything to do with pointers, it is just saying that this is a 2D array of doubles
        Kokkos::View<double **, memorySpace> U("U", NX + 2, NY + 2);
        Kokkos::View<double **, memorySpace> Uprime("Uprime", NX, NY);

        // Initialize the arrays to zero
        iterate(KOKKOS_LAMBDA(int i, int j) { U(i, j) = 0.0; }, 0, 0, NX + 2, NY + 2);
        iterate(KOKKOS_LAMBDA(int i, int j) { Uprime(i, j) = 0.0; }, 0, 0, NX, NY);

        timer t;

        double remin = -2.0, remax = 1.0, immin = -1.5, immax = 1.5;
        int max_iter = 1000;

        double dre = (remax - remin) / NX;
        double dim = (immax - immin) / NY;

        // Apply the Mandelbrot set calculation to each point in the complex plane
        // Probably in a real code this *would* be a functor, but for the sake of demonstrating lambdas we'll just write it inline here
        t.begin("Mandelbrot");
        iterate(KOKKOS_LAMBDA(int i, int j) { 
        double c_re = remin + i * dre;
        double c_im = immin + j * dim;
        double z_re = 0, z_im = 0;
        int iter = 0;

        while (z_re * z_re + z_im * z_im <= 4.0 && iter < max_iter) {
            double z_re_new = z_re * z_re - z_im * z_im + c_re;
            double z_im_new = 2.0 * z_re * z_im + c_im;
            z_re = z_re_new;
            z_im = z_im_new;
            ++iter;
        }
        U(i, j) = iter; }, 0, 0, NX + 2, NY + 2);
        Kokkos::fence();
        t.end();

        iterate(KOKKOS_LAMBDA(int i, int j) { U(i, j) = 0.0; }, 0, 0, NX + 2, NY + 2);
        iterate(KOKKOS_LAMBDA(int i, int j) { Uprime(i, j) = 0.0; }, 0, 0, NX, NY);

        // Apply the Jacobi iteration NITS times
        t.begin("Jacobi");
        for (int i = 0; i < NITS; ++i)
        {
            iterate(KOKKOS_LAMBDA(int i, int j) { Uprime(i, j) = 0.25 * (U(i - 1, j) + U(i + 1, j) + U(i, j - 1) + U(i, j + 1)); }, 1, 1, NX, NY);
            Kokkos::fence();

            iterate(KOKKOS_LAMBDA(int i, int j) { U(i, j) = Uprime(i, j); }, 1, 1, NX, NY);
            Kokkos::fence();
        }
        t.end();
    }
    Kokkos::finalize();
    return 0;
}