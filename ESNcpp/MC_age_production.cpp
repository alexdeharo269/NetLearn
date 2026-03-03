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
double CalculateNARMAError(const vector<vector<double>> &X_states, const vector<double> &predictions, int size_input, double train_ratio);
double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);
double NARMAtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);

int main()
{
    ESNParams params;
    params.reservoir_size = 90;
    params.spectral_radius = 0.99;
    params.input_size = 1;
    params.tau_vals = 10;
    params.output_size = 1;
    params.connectivity_reservoir = 10;

    TaskParams taskparams;
    taskparams.size_input = 1000;
    taskparams.tau_vals = 20;
    taskparams.train_ratio = 0.5;

    // Initialize both tasks once outside the loop
    taskparams.type = MEMORY_CAPACITY;
    Task task_mc(taskparams);

    taskparams.type = NARMA10;
    Task task_narma(taskparams);

    std::ifstream infile("sample_connectomes_with_age.csv");
    std::ofstream outfile("MC_by_age_sample_100.txt");

    vector<vector<vector<double>>> connectomes;
    vector<int> ages;
    tie(connectomes, ages) = process_csv(infile, params.reservoir_size);

    for (int i = 0; i < ages.size(); i++)
    {
        ESN esn_age(params);
        esn_age.setW(connectomes[i]);
        esn_age.rescaleReservoir(params.spectral_radius);

        double MC = MCtest(taskparams, params, esn_age, task_mc);
        double NARMA_error = NARMAtest(taskparams, params, esn_age, task_narma);

        cout << format("\rProcessed: {}/{}", i + 1, ages.size()) << flush;
        outfile << format("{}, {:.4f}, {:.4f}", ages[i], MC, NARMA_error) << endl;
    }
    cout << endl;
    return 0;
}

double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));
    esn.state.assign(params.reservoir_size, 0.0); // Reset state

    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }

    return CalculateMemoryCapacity(X_states, task.predictions, taskparams.tau_vals, taskparams.size_input, taskparams.train_ratio);
}

double NARMAtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));
    esn.state.assign(params.reservoir_size, 0.0); // Reset state

    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }

    return CalculateNARMAError(X_states, task.predictions, taskparams.size_input, taskparams.train_ratio);
}

tuple<vector<vector<vector<double>>>, vector<int>> process_csv(std::ifstream &infile, int size)
{
    vector<vector<vector<double>>> matrices;
    vector<int> ages;
    string line;
    bool first_line = true;

    while (getline(infile, line))
    {
        if (line.empty())
            continue;
        if (first_line)
        {
            first_line = false;
            continue;
        }

        stringstream ss(line);
        string value;
        vector<double> row_vals;

        while (getline(ss, value, ','))
        {
            if (!value.empty())
                row_vals.push_back(stod(value));
        }

        ages.push_back(static_cast<int>(row_vals.back()));

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