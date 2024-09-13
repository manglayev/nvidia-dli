#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

void print_ordinals(const std::vector<unsigned>& ordinals)
{
    std::cout << "ordinals: ";
    for (unsigned ord : ordinals)
        std::cout << std::setw(3) << ord << ' ';
    std::cout << '\n';
}

char to_uppercase(unsigned char c)
{
    return std::toupper(c);
}

void to_uppercase_inplace(char& c)
{
    c = to_uppercase(c);
}

void unary_transform_example(std::string& hello, std::string world)
{
    // Transform string to uppercase in-place

    std::transform(hello.cbegin(), hello.cend(), hello.begin(), to_uppercase);
    std::cout << "hello = " << std::quoted(hello) << '\n';

    // for_each version (see Notes above)
    std::for_each(world.begin(), world.end(), to_uppercase_inplace);
    std::cout << "world = " << std::quoted(world) << '\n';
}

void binary_transform_example(std::vector<unsigned> ordinals)
{
    // Transform numbers to doubled values

    print_ordinals(ordinals);

    std::transform(ordinals.cbegin(), ordinals.cend(), ordinals.cbegin(),
                   ordinals.begin(), std::plus<>{});

    print_ordinals(ordinals);
}

int main()
{
    std::string hello("hello");
    unary_transform_example(hello, "world");

    std::vector<unsigned> ordinals;
    std::copy(hello.cbegin(), hello.cend(), std::back_inserter(ordinals));
    binary_transform_example(std::move(ordinals));
}
