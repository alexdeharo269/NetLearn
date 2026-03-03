#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include <vector>
#include <tuple>
#include "ESN.h"
#include "Task.h"

using namespace std;

tuple<vector<vector<vector<double>>>, vector<int>> process_csv(std::ifstream &infile, int size);
double CalculateClassificationError(const vector<vector<double>> &final_states, const vector<double> &labels, int res_size, double train_ratio);
double ClassificationTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);

int main()
{
    // 1. GLOBAL INPUT (Artificial)
    ESNParams params_glob;
    params_glob.reservoir_size = 90;
    params_glob.spectral_radius = 0.99;
    params_glob.input_size = 1;
    params_glob.output_size = 1;
    params_glob.connectivity_reservoir = 10;
    params_glob.target_nodes = {{}};

    // 2. THALAMIC INPUT (Extreme Bottleneck)
    ESNParams params_thal = params_glob;
    params_thal.target_nodes = {{76, 77}};

    // 3. SENSORY CORTICES INPUT (Distributed Modularity)
    // Occipital (42-53) and Auditory/Superior Temporal (78-81)
    ESNParams params_sensory = params_glob;
    params_sensory.target_nodes = {{42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 78, 79, 80, 81}};

    TaskParams tp;
    tp.train_ratio = 0.5;
    tp.type = CLASSIFICATION;
    Task t_class(tp);

    std::ifstream infile("sample_connectomes_with_age_1000.csv");
    std::ofstream outfile("Classification_Results_All.txt");

    outfile << "Age,Acc_Glob,Acc_Thal,Acc_Sensory\n";

    vector<vector<vector<double>>> connectomes;
    vector<int> ages;
    tie(connectomes, ages) = process_csv(infile, params_glob.reservoir_size);

    for (int i = 0; i < ages.size(); i++)
    {
        ESN esn_glob(params_glob);
        esn_glob.setW(connectomes[i]);
        esn_glob.rescaleReservoir(0.99);
        double err_g = ClassificationTest(tp, params_glob, esn_glob, t_class);

        ESN esn_thal(params_thal);
        esn_thal.setW(connectomes[i]);
        esn_thal.rescaleReservoir(0.99);
        double err_t = ClassificationTest(tp, params_thal, esn_thal, t_class);

        ESN esn_sensory(params_sensory);
        esn_sensory.setW(connectomes[i]);
        esn_sensory.rescaleReservoir(0.99);
        double err_s = ClassificationTest(tp, params_sensory, esn_sensory, t_class);

        cout << format("\rProcessed: {}/{}", i + 1, ages.size()) << flush;
        // Inverting error to accuracy directly in C++ for cleaner Python plotting
        outfile << format("{},{:.4f},{:.4f},{:.4f}", ages[i], 1.0 - err_g, 1.0 - err_t, 1.0 - err_s) << endl;
    }
    cout << endl;
    return 0;
}

double ClassificationTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    int num_samples = task.class_labels.size();
    int res_size = params.reservoir_size;
    vector<vector<double>> final_states(num_samples, vector<double>(res_size, 0.0));

    for (int i = 0; i < num_samples; i++)
    {
        esn.state.assign(res_size, 0.0);
        for (int t = 0; t < task.seq_length; t++)
        {
            esn.input[0] = task.class_sequences[i][t];
            esn.EvoluteReservoir();
        }
        final_states[i] = esn.state;
    }

    return CalculateClassificationError(final_states, task.class_labels, res_size, taskparams.train_ratio);
}
