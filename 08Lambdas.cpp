#include <iostream>
#include <vector>
#include "mdspan/mdspan.hpp"
#include "timer.h"

using arrayType = Kokkos::mdspan<double, Kokkos::extents<std::size_t, Kokkos::dynamic_extent, Kokkos::dynamic_extent>>;


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

    //Initialize the arrays to zero
    iterate([U](int i, int j){ U(i, j) = 0.0; }, 0, 0, NX + 2, NY + 2);
    iterate([Uprime](int i, int j){ Uprime(i, j) = 0.0; }, 0, 0, NX, NY);

    timer t;

    double remin = -2.0, remax = 1.0, immin = -1.5, immax = 1.5;
    int max_iter = 1000;

    double dre = (remax - remin) / NX;
    double dim = (immax - immin) / NY;

    //Apply the Mandelbrot set calculation to each point in the complex plane
    //Probably in a real code this *would* be a functor, but for the sake of demonstrating lambdas we'll just write it inline here
    t.begin("Mandelbrot");
    iterate([U,dre, dim, remin, immin,max_iter](int i, int j){ 
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
        U(i, j) = iter;
    }, 0, 0, NX + 2, NY + 2);
    t.end();

    iterate([U](int i, int j){ U(i, j) = 0.0; }, 0, 0, NX + 2, NY + 2);
    iterate([Uprime](int i, int j){ Uprime(i, j) = 0.0; }, 0, 0, NX, NY);

    //Apply the Jacobi iteration NITS times
    t.begin("Jacobi");
    for(int i = 0; i < NITS; ++i){
        iterate([U, Uprime](int i, int j){
            Uprime(i, j) = 0.25 * (U(i-1, j) + U(i+1, j) + U(i, j-1) + U(i, j+1));
        }, 1, 1, NX, NY);

        iterate([U, Uprime](int i, int j){
            U(i, j) = Uprime(i, j);
        }, 1, 1, NX, NY);
    }
    t.end();
    return 0;
}