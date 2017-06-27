#include "LogisticClassifier.h"
#include "math.h"
using namespace arma;
using namespace std;

LogisticClassifier::LogisticClassifier(mat X, mat Y, double alpha, double thresh , int num_iter)
{
    input_x = addConstCol(X);
    input_y = Y;
    params = randn<mat>(X.n_cols+1,1);
    rate = alpha;
    convergence = thresh;
    max_iterations = num_iter;
}

mat LogisticClassifier::addConstCol(mat X)
{
    X.insert_cols(0,mat(X.n_rows,1,fill::ones));
    return X;
}

double LogisticClassifier::hypo(int i)
{
    double result = accu(input_x.row(i) * params);
    return 1.0/(1+exp(-1.0*result));
}

void LogisticClassifier::gradientDescent()
{
    int counter = 0;
    mat curr = params, prev = params+1;
    float param_change = 100;
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

void LogisticClassifier::train(float ttr)
{
    TTR = ttr;
    gradientDescent();
}

void LogisticClassifier::test()
{
    double cost = 0;
    for(int i=TTR*input_x.n_rows;i<input_x.n_rows;i++)
    {
        cost += input_y(i)*log(hypo(i)) + (1-input_y(i))*log(1-hypo(i));
    }
    cout<<"Avg Cost per Example is : "<<cost/((1-TTR)*input_x.n_rows)<<endl;
}

mat LogisticClassifier::simulate(mat X)
{
    return 1.0/(1+exp(-1.0*(X*params)));
}

void LogisticClassifier::setInputX(mat X){input_x = X;}
void LogisticClassifier::setInputY(mat Y){input_y = Y;}
void LogisticClassifier::setParams(mat B){params = B;}
void LogisticClassifier::setRate(double alpha){rate = alpha;}
void LogisticClassifier::setConvergence(double thresh){convergence = thresh;}
void LogisticClassifier::setIterations(int num_iter){max_iterations = num_iter;}

mat LogisticClassifier::getInputX(){return input_x;}
mat LogisticClassifier::getInputY(){return input_y;}
mat LogisticClassifier::getParams(){return params;}
double LogisticClassifier::getRate(){return rate;}
double LogisticClassifier::getConvergence(){return convergence;}
int LogisticClassifier::getIterations(){return max_iterations;}
