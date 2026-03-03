#include<cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include "ESN.h"
#include "Task.h"

using namespace std;

vector<vector<double>> read_matrix(std::ifstream &infile);
double CalculateMemoryCapacity(const vector<vector<double>> &X_states, const vector<double> &predictions, int tau_vals, int size_input, double train_ratio);
string MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);

int main()
{

    ESNParams params;

    params.reservoir_size = 90; // number of regions in the atlas
    params.spectral_radius = 0.99;
    params.input_size = 1;
    params.tau_vals = 10; // Time lags for memory capacity. We can use a vector if we want to test different taus.
    params.output_size = 1;             // FALTABA
    params.connectivity_reservoir = 10; // FALTABA

    TaskParams taskparams;
    taskparams.size_input = 1000;
    taskparams.tau_vals = 20;
    taskparams.train_ratio = 0.5;

    // 1) Connectome with high disparity
    std::ifstream infile("connectome_high_disparity.dat");

    ESN esn_hdis(params);
    Task taskh(taskparams);

    esn_hdis.setW(read_matrix(infile));
    cout << format("Disparity HDL: {:.4f}", esn_hdis.calculateDisparity(esn_hdis.get_W())) << endl;
    esn_hdis.rescaleReservoir(params.spectral_radius);
    esn_hdis.printSpectralRadius();
    cout << MCtest(taskparams, params, esn_hdis, taskh) << endl;




    // 2) Connectome with low disparity
    std::ifstream infile2("connectome_low_disparity.dat");
    ESN esn_ldis(params);
    Task taskl(taskparams);

    esn_ldis.setW(read_matrix(infile2));
    cout << format("Disparity LDL: {:.4f}", esn_ldis.calculateDisparity(esn_ldis.get_W())) << endl;
    esn_ldis.rescaleReservoir(params.spectral_radius);
    esn_ldis.printSpectralRadius();
    cout << MCtest(taskparams, params, esn_ldis, taskl) << endl;




    // esn.EvoluteReservoir();

    return 0;
}

string MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task){
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));

    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }

    double MC = CalculateMemoryCapacity(X_states, task.predictions, taskparams.tau_vals, taskparams.size_input, taskparams.train_ratio);

    string output = format("Total Memory Capacity: {:.4f}\n", MC);
    return output;
}
