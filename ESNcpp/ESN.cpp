#include "ESN.h"
#include <format>

#include <iostream>
#include <fstream>
#include <sstream>

//Declare functions in Math.cpp so that the compiler knows about them, wihtout needing to include that lib.
double computeSpectralRadius_Internal(const std::vector<std::vector<double>> &matrix);

void ESN::printSpectralRadius() const
{
    // Llamamos a la función "aislada" de Eigen
    double sr = computeSpectralRadius_Internal(W);
    std::cout << std::format("Spectral Radius: {:.4f}\n", sr);
}


void ESN::rescaleReservoir(double spectral_radius_target)
{
    double current_sr = computeSpectralRadius_Internal(W);

    // BLINDAJE: Si Eigen falla o la matriz está vacía, evitamos la división por 0 o NaN
    if (current_sr == 0.0 || std::isnan(current_sr))
    {
        std::cout << "CRITICAL ERROR: Radio Espectral es 0 o NaN. Revisa la matriz." << std::endl;
        return;
    }

    double scale_factor = spectral_radius_target / current_sr;
    for (size_t i = 0; i < W.size(); ++i)
    {
        for (size_t j = 0; j < W[i].size(); ++j)
        {
            W[i][j] *= scale_factor;
        }
    }
}

double ESN::calculateDisparity(const vector<vector<double>> &matrix) const
{
    double disparity = 0.0;
    // Sum of weights per node.
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        double sum_weights = 0.0;
        int sum_neighbors = 0.0;
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            sum_weights += std::abs(matrix[i][j]);
            if (matrix[i][j] != 0.0)
                sum_neighbors++;
        }
        if (sum_neighbors > 0)
        {
            disparity += std::pow(sum_weights / sum_neighbors, 2);
        } // Evitar división por cero
    }

    return disparity;
}

void ESN::printDisparity() const
{
    double disparity = calculateDisparity(W);
    cout << std::format("Disparity: {:.4f}", disparity) << endl;
}
// ESN implementation functions

void ESN::EvoluteReservoir()
{
    // Update reservoir state

    /*
    We fixed the leakage rate $\alpha=1$ following Damicelli et al. (2022) and Suárez et al. (2021).
    This ensures that any memory capacity observed in the reservoir arises exclusively from the reverberations
     within the network topology, rather than from the intrinsic inertia of individual nodes.
    */

    vector<double> new_state(params.reservoir_size, 0.0);
    for (size_t i = 0; i < params.reservoir_size; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < params.reservoir_size; ++j)
        {
            sum += W[i][j] * state[j];
        }
        for (size_t k = 0; k < params.input_size; ++k)
        {
            sum += Win[i * params.input_size + k] * input[k];
        }
        new_state[i] = std::tanh(sum); // Activation function
    }

    // Update state with new_state
    state = new_state;

    // Compute output
    output.resize(params.output_size, 0.0);
    for (size_t i = 0; i < params.output_size; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < params.reservoir_size; ++j)
        {
            sum += Wout[i * params.reservoir_size + j] * state[j];
        }
        output[i] = sum; // Output is linear combination of reservoir states
    }
}
