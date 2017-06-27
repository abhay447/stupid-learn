#include <iostream>
#include "armadillo"
#include "../KMClassifier.h"

using namespace arma;
using namespace std;

void loadInput(mat &X, string input_file)
{
    X.load(input_file);
}

int main()
{
    mat X;
    loadInput(X, "Data/driver_fleet_in.txt");
    KMClassifier KMC(X,4,100,22);
    KMC.classify();
    cout<<"X,Y,Class"<<endl;
    //Generate Output for the CSV to be plotted
    for(int i=0;i<X.n_rows;i++)
    {
        for(int j = 0;j<X.n_cols;j++)
        {
            cout<<X(i,j)<<",";
        }
        cout<<KMC.cluster_points[i]<<endl;
    }
    return 0;
}
