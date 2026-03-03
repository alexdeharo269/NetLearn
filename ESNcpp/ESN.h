#ifndef ESN_H
#define ESN_H

#include <cstdlib>
#include <cmath>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <vector>
#include <math.h>
#include <random>
#include <utility>
#include <limits>
#include<format>


using namespace std;

struct ESNParams
{
    int reservoir_size;     //number of regions in the atlas
    int input_size;         //Usually one input neuron.
    int output_size;        //Depends on the task. 
    double spectral_radius;     //Biggest eigenvalue. Should be less than 1 for the echo state property to hold.
    double sparsesity;          //proportion of non-zero connections in the reservoir. Useful for first initialization.
    int connectivity_reservoir;  //number of connections per neuron in the reservoir. Useful for first initialization.
    double alpha;                // Leaky rate
    double tau_vals;              // Time lags for memory capacity. We can use a vector if we want to test different taus.
    vector<vector<int>> target_nodes={};     // For targeted input: indices of reservoir nodes that receive input.
};

class ESN
{
    // First we need to define a reservoir as a set of global objects Win, W, alpha.
private:
    const ESNParams params;
    vector<double> Win;       // Input weights
    vector<vector<double>> W; // Reservoir weights
    vector<double> Wout;      // Output weights
    double alpha;             // Leaking rate. For now not considered in eqs. set to 1

    vector<double> output;    // Output vector

    inline static mt19937 rng{42}; // Fixed seed for reproducibility

    


    void generateReservoir(const ESNParams &params)
    {
        int size = params.reservoir_size;
        int connectivity_reservoir = params.connectivity_reservoir;
        W.resize(size, vector<double>(size, 0.0));
        int num_connections = size * connectivity_reservoir;

        // Define range of int for random target selection
        uniform_int_distribution<> runint(0, size - 1);
        uniform_real_distribution<> rununif(-1.0, 1.0); // For weight values
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < connectivity_reservoir; ++j)
            {
                W[i][runint(rng)] = rununif(rng);
            }
        }
    }

    void generateInputWeights(const ESNParams &params)
    {
        int size = params.reservoir_size;
        int input_size = params.input_size;
        Win.assign(size * input_size, 0.0);
        uniform_real_distribution<> rununif(-1.0, 1.0); // For weight values
        if (params.target_nodes.empty())
        {
            uniform_real_distribution<> rununif(-1.0, 1.0);
            for (int i = 0; i < size * input_size; ++i)
                Win[i] = rununif(rng);
        }
        else
        {
            for (int k = 0; k < input_size; ++k)
            {
                if (k < params.target_nodes.size())
                {
                    for (int node : params.target_nodes[k])
                    {
                        if (node >= 0 && node < size)
                            Win[node * input_size + k] = rununif(rng);
                    }
                }
            }
        }
    }

    void generateOutput(const ESNParams &params)
    {
        // For now we are going to project in a 10d vector as if we we going to train MNIST.
        // We will do more specialized functions for different tasks.

        Wout.resize(params.output_size*params.reservoir_size, 0.0);
    }

public:
    vector<double> input; // Input vector
    vector<double> state; // Reservoir state vector

    // Constructor: build the ESN. 
    ESN(ESNParams &params) : params(params)
    {
        
        Win.resize(params.reservoir_size * params.input_size, 0.0);
        Wout.resize(params.reservoir_size * params.output_size, 0.0);
        state.assign(params.reservoir_size, 0.0); 
        input.assign(params.input_size, 0.0);
        output.assign(params.output_size, 0.0);

        generateInputWeights(params);
        generateOutput(params);
        generateReservoir(params);
    }
    void setW(const vector<vector<double>> &new_W){W = new_W;}
    const vector<vector<double>> &get_W() const { return W; }
    vector<vector<double>> read_matrix(std::ifstream &infile);

    void rescaleReservoir(double spectral_radius_target);
    //printear spectral radius
    void printSpectralRadius() const;

    double calculateDisparity(const vector<vector<double>> &matrix) const;
    void printDisparity() const;
    

    void EvoluteReservoir();
    
    // For the training we need to define the output weights Wout, and the training function.
    
    // Depending on the task, we can use the trainig error or validation as a measure. Validation is preferred if there
    // is enough data and danger of overfitting, otherwise training error is fine.

    // We can at first do manual parameter selection and then grid search (bigger to smaller grids), and then maybe even random search.
};

#endif
