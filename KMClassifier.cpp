#include "KMClassifier.h"
#include <limits>

using namespace arma;
using namespace std;

KMClassifier::KMClassifier(mat X, int nc, int num_iter, int rand_seed)
{
    input_x = X;
    num_clusters = nc;
    max_iterations = num_iter;
    seed = rand_seed;
    cluster_centroids = zeros<mat> (num_clusters,input_x.n_cols);
    temp_centroids = zeros<mat> (num_clusters,input_x.n_cols);
    randInit();
}

void KMClassifier::randInit()
{
    unordered_set<int> myset;
    srand(seed);
    while(myset.size()<num_clusters)
    {
        int index = (rand() % input_x.n_rows);
        if(myset.find(index) == myset.end())
        {
            cluster_centroids.row(myset.size()) = input_x.row(index);
            myset.insert(index);
        }
    }
    for(int i=0;i<input_x.n_rows;i++)
        cluster_points.push_back(-1);
    
    for(int i=0;i<num_clusters;i++)
        cluster_count.push_back(0);
}

void KMClassifier::assignClass()
{
    for(int i=0;i<num_clusters;i++)
        cluster_count[i] = 0;
    
    for(int i = 0;i<input_x.n_rows;i++)
    {
        double min_dist = std::numeric_limits<double>::max();
        double cluster = -1;
        for(int j=0;j<num_clusters;j++)
        {
            double dist = accu(square(input_x.row(i) - cluster_centroids.row(j)));
            if(dist < min_dist)
            {
                min_dist = dist;
                cluster = j;
            }
        }
        cluster_points[i] = cluster;
        cluster_count[cluster] += 1;
        temp_centroids.row(cluster) += input_x.row(i);
    }
}

void KMClassifier::moveCentroids()
{
    for(int j = 0;j<num_clusters;j++)
    {
        
        cluster_centroids.row(j) = temp_centroids.row(j)/cluster_count[j];
    }
}

void KMClassifier::calcCost()
{
    cost = 0;
    for(int i=0;i<input_x.n_rows;i++)
    {
        cost += accu(square(input_x.row(i) - cluster_centroids.row(cluster_points[i])));
    }
}

void KMClassifier::classify()
{
    for(int i=0;i<max_iterations;i++)
    {
        temp_centroids = zeros<mat> (num_clusters,input_x.n_cols);
        assignClass();
        /*for(int j=0;j<num_clusters;j++)
            cout<<cluster_count[j]<<" ";
        cout<<endl;*/
        moveCentroids();
        calcCost();
    }
}

void KMClassifier::setInputX(mat X){input_x = X;}
mat KMClassifier::getInputX(){return input_x;}