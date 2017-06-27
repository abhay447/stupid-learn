#include <iostream>
#include "armadillo"
#include "../LinearClassifier.h"

using namespace arma;
using namespace std;

void loadInput(mat &X, string input_file, mat &Y, string output_file)
{
    X.load(input_file);
    Y.load(output_file);
}

int main()
{
    mat X,Y;
    loadInput(X, "Data/Linear_in.txt", Y, "Data/Linear_out.txt");
    LinearClassifier LC(X,Y,0.00001);
    LC.train();
    LC.test();
    return 0;
}
