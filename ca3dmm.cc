#include <iostream>
#include <unistd.h>

static void usage(char const* progname) {
    std::cerr << "Usage: " << progname << " n m k -s seeds [-g ge_value] [-v]\n";
}

int main(int argc, char* argv[]) {

    unsigned n = 0, m = 0, k = 0;

    std::string seeds, ge_value;
    bool verbose = false;

    int option;
    // Process command-line arguments using getopt
    while ((option = getopt(argc, argv, "s:g:v")) != -1) {
        switch (option) {
            case 's':
                seeds = optarg;
                break;
            case 'g':
                ge_value = optarg;
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
    if (optind + 2 > argc) {
        usage(argv[0]);
        return 1;
    }

    n = std::stoi(argv[optind++]);
    m = std::stoi(argv[optind++]);
    k = std::stoi(argv[optind++]);

    // Print the parsed values
    std::cout << "n: " << n << '\n';
    std::cout << "m: " << m << '\n';
    std::cout << "k: " << k << '\n';
    std::cout << "seeds: " << seeds << '\n';
    std::cout << "ge_value: " << ge_value << '\n';
    std::cout << "verbose: " << std::boolalpha << verbose << '\n';

    // Rest of your code goes here...

    return 0;
}