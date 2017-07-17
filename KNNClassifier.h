#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include "armadillo"

using namespace arma;
using namespace std;

/*! \brief A class for K-Nearest-Neighbour  Classifier
 *
 *  To classify any occurence we look at the K nearest points in the n-dimensional dataset
 *  and assign the occurence the class that is most common among all k neighbours.
 *
 *  In our approach we use a priority-queue(PQ) of size k. While classifying a point p we scan through all
 *  points in training set. If the size of PQ is less than k or the distance of p from the scanned point is
 *  less than the that of the top point in PQ then we pop the top point from PQ and push the scanned point in PQ.
 *  This ensures the size of PQ to be less than k and that it contains k-nearest-neighbours only.
 */
class KNNClassifier
{
    public:
        //! Constructor for the class.
        /*!
            \param X is the input_x matrix (Dataset)
            \param Y is the observed output matrix (Dataset)
            \param k is the number of nearest neighbours to be considered
        */
        KNNClassifier(mat X, mat Y, int k);
        mat input_x;                                    //!< Input matrix for the dataset
        mat input_y;                                    //!< Output Values for dataset
        int num_nbrs;                                   //!< Number of nearest neighbours to be considered
        void setInputX(mat X);                          //!< setter for input_x
        mat getInputX();                                //!< getter for input_x
        mat classify(mat instance,float ttr=0.75);      //!< Classifies a single observation by looking at \param ttr fraction of inputs
        double test(float ttr=0.75);                    //!< Test the algorithm based on a Train-Test-Ratio = \param ttr

};

/*! \brief A class to keep track of distance of all points of dataset from another point in KNNClassifier
 *  
 *  Every scanned point in the KNN algorithm is treated as an object of this class and 
 *  then stored in a priority-queue(sorted on distance) if required.
 */
class PointDistance
{
    public:
        //! Constructor for the class.
        /*!
            \param i is the index of the scanned point
            \param d is the distance of scanned point from point to be classified
        */
        PointDistance(int i,double d)
        {
            index = i;
            distance = d;
        }
        int index;                                      //!< Index of scanned point
        double distance;                                //!< Distance of scanned point from point to be classified
        bool operator<(const PointDistance &rhs) const  //!< Comparison Function to sort the priority-queue based on distance
        {
            return distance < rhs.distance;
        }
};
#endif
