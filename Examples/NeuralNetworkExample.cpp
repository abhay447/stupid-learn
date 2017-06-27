#include <iostream>
#include "armadillo"
#include "../NeuralNetwork.h"

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
    loadInput(X, "Data/balance_in.txt", Y, "Data/balance_out.txt");
    NeuralNetwork nn(X,Y,0.01,10000);
    nn.addLayer(X.n_cols+1,'o');
    nn.addLayer(6,'h');
    nn.addLayer(Y.n_cols,'o');
    nn.train();
    nn.test();
    return 0;
}
