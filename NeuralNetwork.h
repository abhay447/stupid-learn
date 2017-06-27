#ifndef LINEAR_CLASSIFIER_H
#define LINEAR_CLASSIFIER_H

#include "armadillo"

using namespace arma;
using namespace std;

/*! \brief A Class for layers of the NeuralNetwork.
 *
 *  Each layer has number of units that have their own activations and deltas(error term).
 *  
 *  A layer can be outer(input/ouput) or a hidden layer. The input layer has a bias unit assigned by the network.
 *  The output layer needs no bias unit. The hidden layers have a bias unit by default. This means that if we set a
 *  the number of units of a hidden layer to be 5 then the 5th unit is going to be bias unit.
 *  
 *  A layer has multiple weights from all its units to the all units(except bias) of the next layer.
 *  
 *  The layer class also supports functions like oneHot() and softMax() for better representations of the output. 
 */
class Layer
{
    public:
        //! Constructor for the class.
        /*!
            \param nu is the number of units in the Layer
            \param t is the type of the Layer,'o' is for an input/output layer and 'h' is for a hidden layer
        */
        Layer(int nu, char t='o');
        int num_units;              //!< Number of units of the Layer
        char type;                  //!< Type of the Layer
        mat activations;            //!< activations of all units in the Layer. (Sigmoid for all layers except input)
        mat weights;                //!< weights from all units of current layer to those of next Layer
        mat deltas;                 //!< Delta values for all units(except bias) in Layer.
        void refreshLayer();        //!< Reset all activations(except bias) and deltas to zero 
        mat softMax();              //!< Gives propbability of occurence of a class (Returns a matrix)
        mat oneHot();               //!< Sets most probable class to one and the rest to zero (Returns a matrix)
};



/*! \brief A class for the Neural Network
 *
 *  Neural Network implementation as per "Chapter 4: Artificial Neural Network" in the book "Machine Learning" by "Tom M. Mitchell".
 *  
 *  A neural network here is implemented by using multiple layers as defined in the Layer class.
 *  It takes in as input two matrices. X which has all features of an example in a row and 
 *  Y which has one-hot representation of the classes in a row (check Examples) .
 *
 *  The trained ANN can be used to get the possible activations for each class when fed an input row of feature values.
 */
class NeuralNetwork
{
    public:
        //! Constructor for the class.
        /*!
            \param X is the input_x matrix (feature values in a row)
            \param Y is the input_y matrix (one hot classes in a row)
            \param alpha is the learning rate
            \param num_iter is the maximum number of iterations while training
        */
        NeuralNetwork(mat X, mat Y, double alpha, int num_iter = 30000);
        void train(float ttr = 0.75);           //!< Train the ANN \param ttr is the Train:Test ratio
        void test();                            //!< Test the ANN on the test data as per Train:Test ratio
        mat simulate(mat X);                    //!< Get the most probable class(one hot matrix) for an input
        void setInputX(mat X);                  //!< setter for input_x
        void setInputY(mat Y);                  //!< setter for input_y
        void setRate(double alpha);             //!< setter for learning rate
        void setIterations(int num_iter);       //!< setter for number of iterations
        void addLayer(int num_units, char t);   //!< Add a Layer with num_units units and of type t
        mat getInputX();                        //!< getter for input_x
        mat getInputY();                        //!< getter for input_y
        double getRate();                       //!< getter for learning rate
        int getIterations();                    //!< getter for number of iterations
        
    private:
        mat input_x;                            //!< Feature values of dataset. Each row represnts a training example
        mat input_y;                            //!< Output values(one hot) of dataset. Each row represnts a training example
        mat target;                             //!< Target output for a row of input
        vector<Layer> layers;                   //!< Vector of layers in the Network(1st Layer is input and last layer is output)
        double rate;                            //!< Learning rate
        int max_iterations;                     //!< Maximum number of iterations while training
        float TTR;                              //!< Train:Test ratio to split dataset into training and test set
        mat addConstCol(mat X);                 //!< adds a bias column(of ones) to all rows of input
        void initWeights();                     //!< random initialization of wei
        void refreshNetwork();                  //!< refreshes all layers of the network
        void feedForward();                     //!< Feed-forward
        void backPropagate();                   //!< Backpropagation
};

#endif