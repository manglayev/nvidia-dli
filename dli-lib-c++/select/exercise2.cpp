/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 University of Geneva. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <numeric>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <ranges>
// TODO: add C++ standard library includes as necessary
// #include <...>

// Select elements and copy them to a new vector
template<class UnaryPredicate>
std::vector<int> select(const std::vector<int>& v, UnaryPredicate pred)
{
    // TODO: Allow this version of the code to run in parallel, proceeding in three steps:
    std::vector<char> v_sel(v.size());
    // 1. Fill v_sel with 0/1 values, depending on the outcome of the unary predicatea.

    std::vector<size_t> index(v.size());
    // 2. Compute the cumulative sum of v_sel using inclusive_scan.

    size_t numElem = index.empty() ? 0 : index.back();
    std::vector<int> w(numElem);
    // 3. Use for_each to copy the selected elements from v to w.

    return w;
}

// Initialize vector
void initialize(std::vector<int>& v);

int main(int argc, char* argv[])
{
    // Read CLI arguments, the first argument is the name of the binary:
    if (argc != 2) {
        std::cerr << "ERROR: Missing length argument!" << std::endl;
        return 1;
    }

    // Read length of vector elements
    long long n = std::stoll(argv[1]);

    // Allocate the data vector
    auto v = std::vector<int>(n);

    initialize(v);

    auto predicate = [](int x) { return x % 3 == 0; };
    auto w = select(v, predicate);
    if (!std::all_of(w.begin(), w.end(), predicate) || w.empty()) {
        std::cerr << "ERROR!" << std::endl;
        return 1;
    }
    std::cerr << "OK!" << std::endl;

    std::cout << "w = ";
    std::copy(w.begin(), w.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}

void initialize(std::vector<int>& v)
{
    auto distribution = std::uniform_int_distribution<int> {0, 100};
    auto engine = std::mt19937 {1};
    std::generate(v.begin(), v.end(), [&distribution, &engine]{ return distribution(engine); });
}
