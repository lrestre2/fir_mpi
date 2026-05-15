/* ==========================================================================
 * fir_common.h
 * Shared utilities for the parallel FIR filter MPI project.
 * CPSC-375 -- High-Performance Computing, Spring 2026
 * Liu Restrepo Sanabria
 * ========================================================================== */

#ifndef FIR_COMMON_H
#define FIR_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* --------------------------------------------------------------------------
 * Generate a low-pass FIR filter using a windowed sinc (Hamming window).
 * h      : output array of length N (caller allocates)
 * N      : number of taps (must be odd for symmetric filter)
 * fc     : normalised cutoff frequency (0 < fc < 0.5)
 * -------------------------------------------------------------------------- */
static void generate_fir_coefficients(double *h, int N, double fc)
{
    int M = N - 1;               /* filter order */
    double sum = 0.0;

    for (int k = 0; k < N; k++) {
        double n = k - M / 2.0;
        /* Hamming window */
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * k / M);
        if (n == 0.0)
            h[k] = 2.0 * fc * w;
        else
            h[k] = (sin(2.0 * M_PI * fc * n) / (M_PI * n)) * w;
        sum += h[k];
    }
    /* Normalise to unity DC gain */
    for (int k = 0; k < N; k++)
        h[k] /= sum;
}

/* --------------------------------------------------------------------------
 * Generate a pseudo-random signal in [-1, 1].
 * x      : output array of length L (caller allocates)
 * L      : signal length
 * seed   : RNG seed for reproducibility
 * -------------------------------------------------------------------------- */
static void generate_signal(double *x, long L, unsigned int seed)
{
    srand(seed);
    for (long i = 0; i < L; i++)
        x[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
}

/* --------------------------------------------------------------------------
 * Verify two output arrays element-wise.
 * Returns the maximum absolute error.
 * -------------------------------------------------------------------------- */
static double verify(const double *ref, const double *test, long L, int N)
{
    double max_err = 0.0;
    for (long i = N - 1; i < L; i++) {
        double err = fabs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

#endif /* FIR_COMMON_H */