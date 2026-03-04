#include <iostream>
#include <vector>
#include "mdspan/mdspan.hpp"
#include "timer.h"

using arrayType = Kokkos::mdspan<double, Kokkos::extents<std::size_t, Kokkos::dynamic_extent, Kokkos::dynamic_extent>>;


/**
 * Set all values in the array to zero
 */
struct zeroInit{
    arrayType U;

    zeroInit(arrayType& U_in) : U(U_in) {}

     void operator()(int i, int j){
        U(i, j) = 0.0;
    }
};

/**
 * Apply the Mandelbrot set calculation to each point in the complex plane
 */
struct MandelBrot{
    double re_min, re_max, im_min, im_max;
    double dre, dim;
    int max_iter;
    arrayType output;

    MandelBrot(double re_min_in, double re_max_in, double im_min_in, double im_max_in, int max_iter_in, arrayType& output_in) :
        re_min(re_min_in), re_max(re_max_in), im_min(im_min_in), im_max(im_max_in), max_iter(max_iter_in), output(output_in) {
            dre = (re_max - re_min) / output.extent(0);
            dim = (im_max - im_min) / output.extent(1);
        }

     void operator()(int i, int j){
        double c_re = re_min + i * dre;
        double c_im = im_min + j * dim;
        double z_re = 0, z_im = 0;
        int iter = 0;

        while (z_re * z_re + z_im * z_im <= 4.0 && iter < max_iter) {
            double z_re_new = z_re * z_re - z_im * z_im + c_re;
            double z_im_new = 2.0 * z_re * z_im + c_im;
            z_re = z_re_new;
            z_im = z_im_new;
            ++iter;
        }

        output(i, j) = iter;
    }
};

/**
 * Sweep over the array and apply the Jacobi iteration
 */
struct JacobiSweep{

    arrayType U;
    arrayType Uprime;

    JacobiSweep(arrayType& U_in, arrayType& Uprime_in) : U(U_in), Uprime(Uprime_in) {}

     void operator()(int i, int j){
        Uprime(i, j) = 0.25 * (U(i-1, j) + U(i+1, j) + U(i, j-1) + U(i, j+1));
    }

};

/**
 * Copy the values from Uprime to U
 */
struct Copy{ 
    arrayType U;
    arrayType Uprime;

    Copy(arrayType& U_in, arrayType& Uprime_in) : U(U_in), Uprime(Uprime_in) {}
    
     void operator()(int i, int j){
        U(i, j) = Uprime(i, j);
    }
};

//Iterate over the array and apply the functor to each element
//Use OpenMP to parallelize the outer loop
template<typename T>
void iterate(T functor, int lbound1, int lbound2, int ubound1, int ubound2){
    #pragma omp parallel for schedule(dynamic)
    for(int i = lbound1; i < ubound1; ++i){
        for(int j = lbound2; j < ubound2; ++j){
            functor(i, j);
        }
    }
}


#define NX 4000
#define NY 4000
#define NITS 300

int main(){

    /**
     * Create the arrays. Note that mdspan is just a view onto memory, so we need to create the actual data arrays as well. Use std::vector as a quick and dirty way to do that since it manages memory for us
     */
    std::vector<double> U_data((NX + 2) * (NY + 2), 0.0); //We add 2 to each dimension to account for the ghost cells in the Jacobi iteration
    std::vector<double> Uprime_data(NX * NY, 0.0);

    arrayType U(U_data.data(), NX + 2, NY + 2);
    arrayType Uprime(Uprime_data.data(), NX, NY);

    //Now we create instances of our functors passing the required data to their constructors
    zeroInit initU(U), initUprime(Uprime);
    MandelBrot mandelbrot(-2.0, 1.0, -1.5, 1.5, 1000, U);
    JacobiSweep jacobi(U, Uprime);
    Copy copy(U, Uprime);

    //Initialize the arrays to zero
    iterate(initU, 0, 0, NX + 1, NY + 1);
    iterate(initUprime, 0, 0, NX, NY);

    timer t;

    //Apply the Mandelbrot set calculation to each point in the complex plane
    t.begin("Mandelbrot");
    iterate(mandelbrot, 0,0, NX+1, NY+1);
    t.end();

    //Rezero the arrays before the Jacobi iteration
    iterate(initU, 0, 0, NX + 2, NY + 2);
    iterate(initUprime, 0, 0, NX, NY);

    //Apply the Jacobi iteration NITS times
    t.begin("Jacobi");
    for(int i = 0; i < NITS; ++i){
        iterate(jacobi, 1, 1, NX+1, NY+1);
        iterate(copy, 1, 1, NX+1, NY+1);
    }
    t.end();
    return 0;
}