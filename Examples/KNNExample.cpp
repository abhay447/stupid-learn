#include <iostream>
#include "armadillo"
#include "../KNNClassifier.h"

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
    loadInput(X, "Data/breast-cancer-in.txt", Y, "Data/breast-cancer-out.txt");
    KNNClassifier KNNC(X,Y,11);
    cout<<"Accuracy of the model is : "<<KNNC.test(0.6)<<"%"<<endl;
    return 0;
}
