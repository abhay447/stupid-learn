#include <iostream>
#include "armadillo"
#include "../LogisticClassifier.h"

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
    loadInput(X, "Data/Logistic_in.txt", Y, "Data/Logistic_out.txt");
    LogisticClassifier LC(X,Y,0.1);
    LC.train();
    LC.test();
    cout<<LC.getParams()<<endl;
    return 0;
}
