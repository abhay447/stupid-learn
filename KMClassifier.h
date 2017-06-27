#ifndef KMCLASSIFIER_H
#define KMCLASSIFIER_H

#include <vector>
#include <unordered_set>
#include <cstdlib>
#include "armadillo"

using namespace arma;
using namespace std;

/*! \brief A class for K-Means Classifier
 *  We begin by choosing random class centroids in the dataset.
 *  - Then every point in dataset is assigned a class(cluster) depending on its proximity to the nearest centroid.
 *  - Then every class centroid is moved to the mean location of all points belonging to that class
 *  - The whole process is iterated multiple times with hopes of convergence of centroids
 */
class KMClassifier
{
    public:
        //! Constructor for the class.
        /*!
            \param X is the input_x matrix (Dataset)
            \param nc is number of classes/clusters
            \param num_iter is the number of max iterations
            \param rand_seed is the random seed for a particular run of the experiment.
        */
        KMClassifier(mat X, int nc,int num_iter,int rand_seed);
        mat input_x;                        //!< Input matrix for the dataset
        int max_iterations;                 //!< Max number of iterations of the algorithm
        int num_clusters;                   //!< Number of clusters to classify data into
        int seed;                           //!< Seed for randomly choosing cluster centroids in the begining
        mat cluster_centroids;              //!< Each row represnts the centroid location of a class
        vector<int> cluster_points;         //!< ith entry represents the class assigned to the ith point in dataset
        double cost;                        //!< Sum of distance of all datapoints from assigned centroids
        
        void setInputX(mat X);              //!< setter for input_x
        mat getInputX();                    //!< getter for input_x
        void classify();                    //!< Function to start the iterative classification process
    
    private:
        vector<int> cluster_count;          //!< Tracks number of datapoints in a cluster
        mat temp_centroids;                 //!< Tracks the sum of all datapoints in a cluster.(Helps in shifting centroid)
        
        void assignClass();                 //!< Assign a class to all datapoints based on proximity from centroids
        void moveCentroids();               //!< Move cluster centroids to new cluster mean.
        void calcCost();                    //!< Calculate total cost due to current arrangement of clusters
        void randInit();                    //!< random intialization of cluster centroids
};

#endif
