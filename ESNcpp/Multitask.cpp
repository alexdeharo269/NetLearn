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

// Declaraciones de las funciones de test
double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);
double NARMAtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);
double MultitaskTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);
double GeneralNDTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task);

// Utilidad para extraer hemisferios en el Multitask
vector<vector<double>> mask_states(const vector<vector<double>> &full_states, int start_idx, int end_idx);

int main()
{
    // --- 1. CONFIGURACIÓN 1D (MC, NARMA, TIMESCALE, FAST_PRODUCT) ---
    ESNParams params_1D_global;
    params_1D_global.reservoir_size = 90;
    params_1D_global.spectral_radius = 0.99;
    params_1D_global.input_size = 1;
    params_1D_global.tau_vals = 10;
    params_1D_global.output_size = 1;
    params_1D_global.connectivity_reservoir = 10;
    params_1D_global.target_nodes = {};

    ESNParams params_1D_bio = params_1D_global;
    params_1D_bio.target_nodes = {{76, 77}};

    // --- 2. CONFIGURACIÓN 2D (MULTITASK, HUB) ---
    ESNParams params_2D_global = params_1D_global;
    params_2D_global.input_size = 2;
    params_2D_global.target_nodes = {};

    ESNParams params_2D_bio = params_2D_global;
    params_2D_bio.target_nodes = {{76}, {77}};

    // --- 3. INICIALIZACIÓN DE TAREAS ---
    TaskParams tp;
    tp.size_input = 1000;
    tp.tau_vals = 20;
    tp.train_ratio = 0.5;

    tp.type = MEMORY_CAPACITY;
    Task t_mc(tp);
    tp.type = NARMA10;
    Task t_narma(tp);
    tp.type = MULTITASK;
    Task t_multi(tp);
    tp.type = HUB;
    Task t_hub(tp);
    tp.type = TIMESCALE;
    Task t_time(tp);
    tp.type = FAST_PRODUCT;
    Task t_fast(tp);

    std::ifstream infile("sample_connectomes_with_age_1000.csv");
    std::ofstream outfile("All_Tasks_Results.txt");

    outfile << "Age,MC_Glob,NARMA_Glob,MC_Bio,NARMA_Bio,Multi_Glob,Multi_Bio,Hub_Glob,Hub_Bio,Time_Glob,Time_Bio,Fast_Glob,Fast_Bio\n";

    vector<vector<vector<double>>> connectomes;
    vector<int> ages;
    tie(connectomes, ages) = process_csv(infile, params_1D_global.reservoir_size);

    for (int i = 0; i < ages.size(); i++)
    {
        // 1. Redes Globales
        ESN esn_1D_g(params_1D_global);
        esn_1D_g.setW(connectomes[i]);
        esn_1D_g.rescaleReservoir(0.99);
        ESN esn_2D_g(params_2D_global);
        esn_2D_g.setW(connectomes[i]);
        esn_2D_g.rescaleReservoir(0.99);

        double mc_g = MCtest(tp, params_1D_global, esn_1D_g, t_mc);
        double nar_g = NARMAtest(tp, params_1D_global, esn_1D_g, t_narma);
        double time_g = GeneralNDTest(tp, params_1D_global, esn_1D_g, t_time);
        double fast_g = GeneralNDTest(tp, params_1D_global, esn_1D_g, t_fast); // NUEVO
        double multi_g = MultitaskTest(tp, params_2D_global, esn_2D_g, t_multi);
        double hub_g = GeneralNDTest(tp, params_2D_global, esn_2D_g, t_hub);

        // 2. Redes Biológicas
        ESN esn_1D_b(params_1D_bio);
        esn_1D_b.setW(connectomes[i]);
        esn_1D_b.rescaleReservoir(0.99);
        ESN esn_2D_b(params_2D_bio);
        esn_2D_b.setW(connectomes[i]);
        esn_2D_b.rescaleReservoir(0.99);

        double mc_b = MCtest(tp, params_1D_bio, esn_1D_b, t_mc);
        double nar_b = NARMAtest(tp, params_1D_bio, esn_1D_b, t_narma);
        double time_b = GeneralNDTest(tp, params_1D_bio, esn_1D_b, t_time);
        double fast_b = GeneralNDTest(tp, params_1D_bio, esn_1D_b, t_fast); // NUEVO
        double multi_b = MultitaskTest(tp, params_2D_bio, esn_2D_b, t_multi);
        double hub_b = GeneralNDTest(tp, params_2D_bio, esn_2D_b, t_hub);

        cout << format("\rProcessed: {}/{}", i + 1, ages.size()) << flush;
        outfile << format("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}",
                          ages[i], mc_g, nar_g, mc_b, nar_b, multi_g, multi_b, hub_g, hub_b, time_g, time_b, fast_g, fast_b)
                << endl;
    }
    cout << endl;
    return 0;
}

// ================= TEST FUNCTIONS =================

double MCtest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));
    esn.state.assign(params.reservoir_size, 0.0);
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
    esn.state.assign(params.reservoir_size, 0.0);
    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }
    return CalculateNARMAError(X_states, task.predictions, taskparams.size_input, taskparams.train_ratio);
}

double MultitaskTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> full_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));
    esn.state.assign(params.reservoir_size, 0.0);
    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.inputs[0][t];
        esn.input[1] = task.inputs[1][t];
        esn.EvoluteReservoir();
        full_states[t] = esn.state;
    }
    // Segregación espacial estricta (Hemisferio Izquierdo 0-45, Derecho 45-90)
    vector<vector<double>> X_left = mask_states(full_states, 0, 45);
    vector<vector<double>> X_right = mask_states(full_states, 45, 90);

    double err_L = CalculateNARMAError(X_left, task.targets[0], taskparams.size_input, taskparams.train_ratio);
    double err_R = CalculateNARMAError(X_right, task.targets[1], taskparams.size_input, taskparams.train_ratio);
    return (err_L + err_R) / 2.0;
}

// Función genérica para Hub y Timescale (que solo tienen 1 target y leen de todo el cerebro)
double GeneralNDTest(TaskParams &taskparams, ESNParams &params, ESN &esn, Task &task)
{
    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));
    esn.state.assign(params.reservoir_size, 0.0);
    for (int t = 0; t < taskparams.size_input; t++)
    {
        for (int k = 0; k < params.input_size; k++)
            esn.input[k] = task.inputs[k][t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }
    return CalculateNARMAError(X_states, task.targets[0], taskparams.size_input, taskparams.train_ratio);
}
