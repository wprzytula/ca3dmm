#include "densematgen.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <vector>

constexpr double const l = 0.95;

namespace {
struct Config
{
    int n, m, k, p, p_n, p_m, p_k, rank;

    Config(int const n, int const m, int const k, int const p, int const rank)
        : n{n}, m{m}, k{k}, p{p}, rank{rank}
    {
        // Solve min(p_m*k*n + p_n*m*k + p_k*m*n) with constraints:
        // (a) l*p ≤ p_m*p_n*p_k ≤ p (l is a constant, e.g. 0.95; this ensures we
        // use
        //     as many processors as possible, but not necessarily all processors
        //     available)
        // (b) mod(max(p_m, p_n), min(p_m, p_n)) == 0 (this ensures the Cannon
        // algorithm can be performed). Among admissible, optimal solutions, pick
        // the one using as many processors as possible, thus maximizing
        // p_m*p_n*p_k.
        int opt_p_n = p;
        int opt_p_m = 1;
        int opt_p_k = 1;
        int max_p_prod = opt_p_m * opt_p_n * opt_p_k;
        int min_sum = opt_p_m * k * n + opt_p_n * m * k + opt_p_k * m * n;

        int const l_p = l * p;

        for (int p_m = 1; p_m < p; ++p_m)
        {
            if (p_m * p_m > p)
                break; // No point in continuing if p_m * p_m exceeds p
            int const p_div_p_m = p / p_m;
            for (int p_n = p_m; p_n < p_div_p_m; p_n += p_m)
            {
                int const p_k = p / (p_m * p_n);
                int const p_prod = p_k * p_m * p_n;
                int const sum = p_m * k * n + p_n * m * k + p_k * m * n;
                if (l_p <= p_prod && p_prod <= p &&
                    (sum < min_sum || (sum == min_sum && p_prod > max_p_prod)))
                {
                    min_sum = sum;
                    max_p_prod = p_prod;
                    opt_p_n = p_n;
                    opt_p_m = p_m;
                    opt_p_k = p_k;
                }
            }
        }
        p_n = opt_p_n;
        p_m = opt_p_m;
        p_k = opt_p_k;
    }

    void print() const
    {
        printf("CONFIG: n=%i, m=%i, k=%i, p=%i ---> p_m=%i, p_n=%i, p_k=%i (prod: "
               "%i, sum: %i)\n",
               n, m, k, p, p_m, p_n, p_k, p_n * p_m * p_k,
               p_m * k * n + p_n * m * k + p_k * m * n);
    }

    static void generate_matrix(int const i, int const j, int const seed)
    {
        for (int r = 0; r < i; r++)
        {
            for (int c = 0; c < j; c++)
            {
                const double entry = generate_double(seed, r, c);
                std::cout << entry << " ";
            }
            std::cout << std::endl;
        }
    }

    void multiply(int const seed_a, int const seed_b, bool const verbose,
                  bool const ge, double const ge_value) const
    {
    // Algorithm:
    // 2. Organize processes into pk groups, each group has pm×pn processes.
    // 3. Assign A to pk groups in a block-column order; assign B to pk groups in a block-row order.
    //    Denote by Ai the block-column of A assigned to group i; and by Bi the block-row of B assigned to group i.
    // 4. Within the i-th group: organize processes into c=max(pm,pn)/min(pm,pn) Cannon groups. For example,
    // assume pn>pm. Bi is then not replicated. Ai is replicated c times — each Cannon group stores its complete copy of Ai.
    // 5. Perform the Cannon's algorithm in each Cannon group and in each of the pk groups to get Ci.
    // 6. Reduce C=∑ki=1Ci.


    // 2. Organize processes into pk groups, each group has pm×pn processes.

        // matrix A
        generate_matrix(m, k, seed_a);
        generate_matrix(k, n, seed_b);
    }

};

static void usage(char const *progname)
{
    std::cerr << "Usage: " << progname << " n m k -s seeds [-g ge_value] [-v]\n";
}
}

#ifdef TEST
// UNIT TESTS
int main()
{
    Config const conf{323, 123, 231, 1023, 0};
    conf.print();
    return 0;
}

#else
int main(int argc, char *argv[])
{
    int p = 0;
    int rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    static char const *const delim = ",";

    int n = 0, m = 0, k = 0;

    char *seeds = nullptr, *ge_value_str = nullptr;
    bool verbose = false;

    int option;
    // Process command-line arguments using getopt
    while ((option = getopt(argc, argv, "s:g:v")) != -1)
    {
        switch (option)
        {
        case 's':
            seeds = optarg;
            break;
        case 'g':
            ge_value_str = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // Parse remaining positional arguments n, m, and k
    if (optind + 2 > argc)
    {
        usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    n = std::stoi(argv[optind++]);
    m = std::stoi(argv[optind++]);
    k = std::stoi(argv[optind++]);

    double const ge_value = ge_value_str ? std::stod(ge_value_str) : 0.;

    Config const conf{n, m, k, p, rank};

    // Print the parsed values
    std::cout << "n: " << n << '\n';
    std::cout << "m: " << m << '\n';
    std::cout << "k: " << k << '\n';
    std::cout << "seeds: " << seeds << '\n';
    std::cout << "ge_value: " << ge_value << '\n';
    std::cout << "verbose: " << std::boolalpha << verbose << '\n';

    char *token = std::strtok(seeds, delim);

    while (token != nullptr)
    {
        int first = std::stoi(token);

        token = std::strtok(nullptr, delim);
        if (token != nullptr)
        {
            int second = std::stoi(token);

            std::cout << "Pair: " << first << delim << second << std::endl;
            // Rest of your code goes here...
            conf.multiply(first, second, verbose, !!ge_value_str, ge_value);

            token = std::strtok(nullptr, delim);
        }
        else
        {
            std::cerr << "Odd number of seeds. Unpaired: " << first << '\n';
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
#endif // TEST