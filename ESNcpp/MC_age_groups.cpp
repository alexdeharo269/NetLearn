#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include "ESN.h"
#include "Task.h"

using namespace std;

tuple<vector<vector<vector<double>>>, vector<int>> process_csv(std::ifstream &infile, int size);
double CalculateMemoryCapacity(const vector<vector<double>> &X_states, const vector<double> &predictions, int tau_vals, int size_input, double train_ratio);
double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);

int main()
{

    ESNParams params;

    params.reservoir_size = 90; // number of regions in the atlas
    params.spectral_radius = 0.99;
    params.input_size = 1;
    params.tau_vals = 10;               // Time lags for memory capacity. We can use a vector if we want to test different taus.
    params.output_size = 1;             // FALTABA
    params.connectivity_reservoir = 10; // FALTABA

    TaskParams taskparams;
    taskparams.type = MEMORY_CAPACITY;

    taskparams.size_input = 1000;
    taskparams.tau_vals = 20;
    taskparams.train_ratio = 0.5;

    Task task(taskparams);

    std::ifstream infile("sample_connectomes_with_age_1000.csv");
    std::ofstream outfile("MC_by_age_sample_1000.txt");

    // esn.EvoluteReservoir();
    vector<vector<vector<double>>> connectomes;
    vector<int> ages;
    tie(connectomes, ages) = process_csv(infile, params.reservoir_size);

    for (int i=0; i<ages.size(); i++)
    {
        ESN esn_age(params);
        esn_age.setW(connectomes[i]);
        esn_age.rescaleReservoir(params.spectral_radius);
        double mc = MCtest(taskparams, params, esn_age, task);
        //cout << format("Age: {}, Memory Capacity: {:.4f}", ages[i], mc) << endl;
        cout << format("\rProcessed: {}/{}", i+1, ages.size()) << flush;
        outfile << format("{}, {:.4f}", ages[i], mc) << endl;
    }
    cout << endl;
    return 0;
}

double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));

    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }

    double MC = CalculateMemoryCapacity(X_states, task.predictions, taskparams.tau_vals, taskparams.size_input, taskparams.train_ratio);

    
    return MC;
}
tuple<vector<vector<vector<double>>>, vector<int>> process_csv(std::ifstream &infile, int size)
{
    vector<vector<vector<double>>> matrices;
    vector<int> ages;
    string line;

    // Skip the header if your CSV has one. If not, remove this boolean flag.
    bool first_line = true;

    while (getline(infile, line))
    {
        

        stringstream ss(line);
        string value;
        vector<double> row_vals;

        // 1. Read all comma-separated values in the row
        while (getline(ss, value, ','))
        {
            if (!value.empty())
            {
                row_vals.push_back(stod(value));
            }
        }

        // 2. The last value is the age group
        ages.push_back(static_cast<int>(row_vals.back()));

        // 3. Reconstruct the size x size (90x90) matrix from the remaining values
        vector<vector<double>> matrix(size, vector<double>(size, 0.0));
        int flat_index = 0;
        for (int r = 0; r < size; ++r)
        {
            for (int c = 0; c < size; ++c)
            {
                matrix[r][c] = row_vals[flat_index++];
            }
        }

        matrices.push_back(matrix);
    }

    return make_tuple(matrices, ages);
}
