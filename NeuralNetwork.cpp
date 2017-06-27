#include "NeuralNetwork.h"
#include "math.h"
using namespace arma;
using namespace std;


NeuralNetwork::NeuralNetwork(mat X, mat Y, double alpha, int num_iter)
{
    input_x = addConstCol(X);
    input_y = Y;
    rate = alpha;
    max_iterations = num_iter;
}

void NeuralNetwork::refreshNetwork()
{
    for(int l=0;l<layers.size();l++)
    {
        layers[l].refreshLayer();
    }
}

mat NeuralNetwork::addConstCol(mat X)
{
    X.insert_cols(0,mat(X.n_rows,1,fill::ones));
    return X;
}

void NeuralNetwork::addLayer(int num_units, char t)
{
    Layer l(num_units,t);
    layers.push_back(l);
}

void NeuralNetwork::initWeights()
{
    for(int l=0;l<layers.size()-1;l++)
    {
        layers[l].weights = randn<mat>(layers[l].num_units,layers[l+1].num_units);
        if(layers[l+1].type == 'h')
        {
            layers[l].weights.col(layers[l+1].num_units-1) = zeros<mat>(layers[l].num_units,1);
        }
    }
}

void NeuralNetwork::feedForward()
{
    for(int l=0;l<layers.size()-1;l++)
    {
        mat act = layers[l].weights.t() * layers[l].activations;
        layers[l+1].activations = 1.0/(1+exp(-1.0*act));
    }   
}

void NeuralNetwork::backPropagate()
{
    int last = layers.size()-1;
    for(int i = 0;i<layers[last].num_units;i++)
    {
        layers[last].deltas[i] = (target[i] - layers[last].activations[i])\
            * layers[last].activations[i] * (1 - layers[last].activations[i]);
    }
    for(int l=layers.size()-2;l>=0;l--)
    {
        int r = layers[l].type == 'h'?layers[l].num_units-1:layers[l].num_units;
        int c = layers[l+1].type == 'h'?layers[l+1].num_units-1:layers[l+1].num_units;
        for(int i = 0;i<r;i++)
        {            
            double delta = 0;
            for(int j = 0;j<c;j++)
            {
                delta += layers[l].weights(i,j) * layers[l+1].deltas[j];
            }
            layers[l].deltas[i] = delta * layers[l].activations[i] * (1 - layers[l].activations[i]);            
        }
    }
    
    for(int l=layers.size()-1;l>=0;l--)
    {
        int r = layers[l].type == 'h'?layers[l].num_units-1:layers[l].num_units;
        int c = layers[l+1].type == 'h'?layers[l+1].num_units-1:layers[l+1].num_units;
        for(int i = 0;i<r;i++)
        {
            for(int j = 0;j<c;j++)
            {
                
                layers[l].weights(i,j) += rate * layers[l+1].deltas[j] * layers[l].activations[i];
            }
        }
    }
}

void NeuralNetwork::train(float ttr)
{
    TTR = ttr;
    initWeights();
    for(int t=0;t<max_iterations;t++)
    {
        for(int i=0;i<TTR*input_x.n_rows;i++)
        {
            layers[0].activations = input_x.row(i).t();
            target = input_y.row(i).t();
            feedForward();
            backPropagate();
        }
        refreshNetwork();
        cout<<'\r'<<t<<" turns have passed "<<flush;
    }
    cout<<endl;
}

void NeuralNetwork::test()
{
    float error = 0;
    for(int i=TTR*input_x.n_rows;i<input_x.n_rows;i++)
    {
        layers[0].activations = input_x.row(i).t();
        target = input_y.row(i).t();
        feedForward();
        mat one_hot = layers.back().oneHot();
        error += accu(abs(target-one_hot));
    }
    cout<<"Accuracy : "<<100.0*(1-error/(1.0 - TTR)/input_x.n_rows)<<"%"<<endl;
}

mat NeuralNetwork::simulate(mat X)
{
    X = addConstCol(X);
    layers[0].activations = X.t();
    feedForward();
    return layers.back().activations;
}

Layer::Layer(int nu,char t)
{
    num_units = nu;
    type = t;
    activations = zeros<mat>(num_units,1);
    if(type == 'h')
        activations[num_units-1] = 1;
    deltas = zeros<mat>(num_units,1);
}

void Layer::refreshLayer()
{
    activations = zeros<mat>(num_units,1);
    if(type == 'h')
        activations[num_units-1] = 1;
    deltas = zeros<mat>(num_units,1);
}

mat Layer::softMax()
{
    mat sm = exp(activations);
    sm /= accu(sm);
    return sm;
}

mat Layer::oneHot()
{
    int max_index = 0;
    float max = 0;
    for(int t=0;t<activations.n_rows;t++)
    {
        if(max < activations[t])
        {
           max = activations[t];
           max_index = t;
        }        
    }
    mat act = zeros<mat>(num_units,1);
    act[max_index] = 1;
    return act;
}

void NeuralNetwork::setInputX(mat X){input_x = X;}
void NeuralNetwork::setInputY(mat Y){input_y = Y;}
void NeuralNetwork::setRate(double alpha){rate = alpha;}
void NeuralNetwork::setIterations(int num_iter){max_iterations = num_iter;}

mat NeuralNetwork::getInputX(){return input_x;}
mat NeuralNetwork::getInputY(){return input_y;}
double NeuralNetwork::getRate(){return rate;}
int NeuralNetwork::getIterations(){return max_iterations;}
