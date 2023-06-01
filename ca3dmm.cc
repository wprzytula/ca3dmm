#include "densematgen.h"
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

static void usage(char const *progname)
{
    std::cerr << "Usage: " << progname << " n m k -s seeds [-g ge_value] [-v]\n";
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

static void multiply(int const m, int const n, int const k, int const seed_a,
                     int const seed_b, bool const verbose, bool const ge,
                     double const ge_value)
{
    // matrix A
    generate_matrix(m, k, seed_a);
    generate_matrix(k, n, seed_b);
}

int main(int argc, char *argv[])
{
    static char const *const delim = ",";

    unsigned n = 0, m = 0, k = 0;

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
            return 1;
        }
    }

    // Parse remaining positional arguments n, m, and k
    if (optind + 2 > argc)
    {
        usage(argv[0]);
        return 1;
    }

    n = std::stoi(argv[optind++]);
    m = std::stoi(argv[optind++]);
    k = std::stoi(argv[optind++]);

    double const ge_value = ge_value_str ? std::stod(ge_value_str) : 0.;

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
            multiply(m, n, k, first, second, verbose, !!ge_value_str, ge_value);

            token = std::strtok(nullptr, delim);
        }
        else
        {
            std::cerr << "Odd number of seeds. Unpaired: " << first << '\n';
            return 1;
        }
    }

    return 0;
}