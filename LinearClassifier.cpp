#include "LinearClassifier.h"
using namespace arma;
using namespace std;

LinearClassifier::LinearClassifier(mat X, mat Y, double alpha, double thresh , int num_iter)
{
    input_x = addConstCol(X);
    input_y = Y;
    params = randn<mat>(X.n_cols+1,1);
    rate = alpha;
    convergence = thresh;
    max_iterations = num_iter;
}

mat LinearClassifier::addConstCol(mat X)
{
    X.insert_cols(0,mat(X.n_rows,1,fill::ones));
    return X;
}

double LinearClassifier::hypo(int i)
{
    return accu(input_x.row(i) * params);;
}

void LinearClassifier::gradientDescent()
{
    int counter = 0;
    mat curr = params, prev = params+1;
    double param_change = 100;
    while(counter<max_iterations && param_change > convergence)
    {
        vector<double> updates;
        for(int j=0;j<params.n_rows;j++)
        {
            double cost_derivative = 0;
            for(int i=0;i<TTR*input_x.n_rows;i++)
            {
                cost_derivative += (hypo(i) - input_y(i))*input_x(i,j);
            }
            updates.push_back(cost_derivative);
        }
        for(int j=0;j<params.n_rows;j++)
        {
             params(j,0) = params(j,0) - rate * updates[j]/input_x.n_rows/TTR;            
        }
        counter++;
        prev = curr;
        curr = params;
        param_change = 100 * accu(abs(curr - prev))/accu(abs(curr));
        cout<<'\r'<<counter<<" turns have passed with "<<param_change<<"% change in parameters "<<flush;
    }
    cout<<endl;
}

void LinearClassifier::train(float ttr)
{
    TTR = 0.75;
    gradientDescent();
}

void LinearClassifier::test()
{
    float cost = 0;
    for(int i=TTR*input_x.n_rows;i<input_x.n_rows;i++)
    {
        cost += 0.5*accu(square(input_y(i) - simulate(input_x.row(i))));
    }
    cout<<"Avg cost per example = "<<cost/((1-TTR)*input_x.n_rows)<<endl;
}

mat LinearClassifier::simulate(mat X)
{
    return X*params;    
}

void LinearClassifier::setInputX(mat X){input_x = X;}
void LinearClassifier::setInputY(mat Y){input_y = Y;}
void LinearClassifier::setParams(mat B){params = B;}
void LinearClassifier::setRate(double alpha){rate = alpha;}
void LinearClassifier::setConvergence(double thresh){convergence = thresh;}
void LinearClassifier::setIterations(int num_iter){max_iterations = num_iter;}

mat LinearClassifier::getInputX(){return input_x;}
mat LinearClassifier::getInputY(){return input_y;}
mat LinearClassifier::getParams(){return params;}
double LinearClassifier::getRate(){return rate;}
double LinearClassifier::getConvergence(){return convergence;}
int LinearClassifier::getIterations(){return max_iterations;}
