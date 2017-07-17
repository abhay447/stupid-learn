#include "KNNClassifier.h"
#include <limits>
#include <queue>

using namespace arma;
using namespace std;

KNNClassifier::KNNClassifier(mat X, mat Y, int k)
{
    input_x = X;
    input_y = Y;
    num_nbrs = k;
}

void oneHot(mat &m)
{
    int max_index = 0;
    float max = 0;
    for(int t=0;t<m.n_cols;t++)
    {
        if(max < m(0,t))
        {
           max = m(0,t);
           max_index = t;
        }        
    }
    m = zeros<mat>(1,m.n_cols);
    m(0,max_index) = 1;
}

mat KNNClassifier::classify(mat instance,float ttr)
{
    priority_queue<PointDistance> nbrs;    
    for(int i=0;i<input_x.n_rows*ttr;i++)
    {
        PointDistance p(i,accu(square(input_x.row(i) - instance)));
        if(nbrs.size() == num_nbrs)
        {
            if(nbrs.top().distance > p.distance)
            {
                nbrs.pop();
                nbrs.push(p);
            }
        }
        else
            nbrs.push(p);
    }
    
    mat result = zeros<mat>(1,input_y.n_cols);
    while(nbrs.size()>0)
    {
        result += input_y.row(nbrs.top().index);
        nbrs.pop();
    }
    //result /= num_nbrs;
    oneHot(result);
    return result;
}

double KNNClassifier::test(float ttr)
{
    int mishap = 0;
    for(int i=input_x.n_rows*ttr;i<input_x.n_rows;i++)
    {
        mat error = classify(input_x.row(i),ttr)-input_y.row(i);
        if(accu(square(error))!=0)
            mishap++;
    }
    return 100.0 - mishap*100.0/(input_x.n_rows*(1-ttr));
}

void KNNClassifier::setInputX(mat X){input_x = X;}
mat KNNClassifier::getInputX(){return input_x;}