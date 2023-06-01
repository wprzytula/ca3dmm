/*
 * Matrix generation library for the MPI assignment, HPC, Spring 2022/23.
 *
 * Faculty of Mathematics, Informatics and Mechanics.
 * University of Warsaw, Warsaw, Poland.
 *
 * Krzysztof Rzadca
 * LGPL, 2023
 */

#ifndef __MIMUW_MATGEN_H__
#define __MIMUW_MATGEN_H__

/**
 * Generates a single matrix entry at (row, col).
 * 
 * Stateless generator
 * @param seed seed for the generator (some seeds switch op mode).
 * @param row row coordinate of the generated element.
 * @param col col coordinate of the generated element.
 * @return A very-pseudo random matrix element.
 */
double generate_double(int seed, int row, int col);

#endif /* __MIMUW_MATGEN_H__ */
