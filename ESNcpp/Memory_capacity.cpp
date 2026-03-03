// depende de la tarea. Si fuera MNITS, seria 28^2=784. Podemos tomar 1 como ejemplo.
// proyeccion aleatoria las regiones en nuestro caso que solo tenemos conectomas.
// en el caso de LE Suarez, serían las regiones subcorticales.
//  en el de de damicelli, hacen lo que nosotros (proyectar aleatorios entre -1 y 1)
#include <cstdlib>
#include <iostream>
#include <vector>
#include <format>
#include "ESN.h"
#include "Task.h"

using namespace std;

// Forward declaration
double CalculateMemoryCapacity(const vector<vector<double>> &X_states, const vector<double> &predictions, int tau_vals, int size_input, double train_ratio);

int main()
{
    ESNParams params;
    params.reservoir_size = 20;
    params.spectral_radius = 0.99;
    params.sparsesity = 0.1;
    params.connectivity_reservoir = 10;
    params.input_size = 1;
    params.output_size = 1;

    TaskParams taskparams;
    taskparams.size_input = 1000;
    taskparams.tau_vals = 20;
    taskparams.train_ratio = 0.5;

    Task task(taskparams);

    ESN esn(params);
    esn.rescaleReservoir(params.spectral_radius);
    
    esn.printSpectralRadius();

    vector<vector<double>> X_states(taskparams.size_input, vector<double>(params.reservoir_size, 0.0));

    for (int t = 0; t < taskparams.size_input; t++)
    {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        X_states[t] = esn.state;
    }
    

    double MC = CalculateMemoryCapacity(X_states, task.predictions, taskparams.tau_vals, taskparams.size_input, taskparams.train_ratio);

    cout << format("Total Memory Capacity: {:.4f}\n", MC);

    return 0;
}