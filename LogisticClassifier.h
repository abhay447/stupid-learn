#ifndef LOGISTIC_CLASSIFIER_H
#define LOGISTIC_CLASSIFIER_H

#include "armadillo"

using namespace arma;
using namespace std;

/*! \brief A Class for Logistic Regression.
 *
 *  Output is a sigmoid of linear weighted combination of parameters.
 *  The process is terminated on convergence of weights or after a fixed number of iterations.
 */
class LogisticClassifier
{
    public:
        //! Constructor for the class.
        /*!
            \param X is the input_x matrix (feature values in a row)
            \param Y is the input_y matrix (actual out for the input row)
            \param alpha is the learning rate
            \param thresh is the parameter to check the convergence of weights
            \param num_iter is the maximum number of iterations while training
        */
        LogisticClassifier(mat X, mat Y, double alpha, double thresh = 0.06, int num_iter = 300000);
        void train(float ttr = 0.75);               //!< Train the ANN \param ttr is the Train:Test ratio
        void test();                                //!< Test the ANN on the test data as per Train:Test ratio
        mat simulate(mat X);                        //!< Predict output value for a new row of input
        void setInputX(mat X);                      //!< setter for input_x
        void setInputY(mat Y);                      //!< setter for input_y
        void setParams(mat B);                      //!< setter for params(weights)
        void setRate(double alpha);                 //!< setter for learning rate
        void setConvergence(double thresh);         //!< setter for convergence threshold of weights
        void setIterations(int num_iter);           //!< setter for max number of iterations
        mat getInputX();                            //!< getter for input_x
        mat getInputY();                            //!< getter for input_y
        mat getParams();                            //!< getter for weights
        double getRate();                           //!< getter for learning rate
        double getConvergence();                    //!< getter for convergence threshold
        int getIterations();                        //!< getter for max number of iterations
        
    private:
        mat input_x;                                //!< Feature values of dataset. Each row represnts a training example
        mat input_y;                                //!< Output values of dataset. Each row represnts a training example
        mat params;                                 //!< Row vector of weights from all features to output
        double rate;                                //!< Learning rate
        double convergence;                         //!< Convergence threshold for weights
        int max_iterations;                         //!< Maximum number of iterations while training
        float TTR;                                  //!< Train:Test ratio to split dataset into training and test set
        mat addConstCol(mat X);                     //!< adds a bias column(of ones) to all rows of input
        double hypo(int i);                         //!< Hypothesis Function
        void gradientDescent();                     //!< Gradient Descent
};

#endif