#ifndef TASK_H
#define TASK_H

#include <random>
#include <vector>
#include <cstdlib>

using namespace std;

enum TaskType
{
    MEMORY_CAPACITY,
    NARMA10,
    MULTITASK,
    HUB,
    TIMESCALE,
    FAST_PRODUCT,
    CLASSIFICATION
};

struct TaskParams
{
    TaskType type;
    int tau_vals;
    int size_input;
    double train_ratio;
};

class Task
{
public:
    const TaskParams params;

    // Variables para tareas 1D (MC, NARMA)
    vector<double> sequence;
    vector<double> predictions;

    // Variables para tareas multidimensionales (Multitask, Hub, Timescale)
    vector<vector<double>> inputs;  // [input_dim][time]
    vector<vector<double>> targets; // [target_dim][time]

    // Variables para tarea de Clasificación
    int num_samples = 200;
    int seq_length = 50;
    vector<vector<double>> class_sequences; // [sample][time]
    vector<double> class_labels;            // [sample]


private:
    inline static mt19937 rng{42};

    void generateMemoryCapacity()
    {
        sequence.resize(params.size_input);
        predictions.assign(params.tau_vals * params.size_input, 0.0);
        uniform_real_distribution<> dist(-1.0, 1.0);

        for (int i = 0; i < params.size_input; i++)
            sequence[i] = dist(rng);

        for (int i = 0; i < params.tau_vals; i++)
        {
            int tau = i + 1;
            for (int j = tau; j < params.size_input; j++)
            {
                predictions[i * params.size_input + j] = sequence[j - tau];
            }
        }
    }

    void generateNARMA10()
    {
        sequence.resize(params.size_input);
        predictions.assign(params.size_input, 0.0);
        uniform_real_distribution<> dist(0.0, 0.5);

        for (int i = 0; i < params.size_input; i++)
            sequence[i] = dist(rng);

        for (int t = 9; t < params.size_input - 1; t++)
        {
            double sum_y = 0.0;
            for (int i = 0; i < 10; i++)
                sum_y += predictions[t - i];

            predictions[t + 1] = 0.3 * predictions[t] +
                                 0.05 * predictions[t] * sum_y +
                                 1.5 * sequence[t - 9] * sequence[t] + 0.1;
        }
    }

    void generateMultitask()
    {
        inputs.assign(2, vector<double>(params.size_input, 0.0));
        targets.assign(2, vector<double>(params.size_input, 0.0));
        uniform_real_distribution<> dist(-1.0, 1.0);

        for (int t = 0; t < params.size_input; t++)
        {
            inputs[0][t] = dist(rng);
            inputs[1][t] = dist(rng);
        }
        int delay = 5; // Retraso fijo para evaluar la interferencia espacial
        for (int t = delay; t < params.size_input; t++)
        {
            targets[0][t] = inputs[0][t - delay];
            targets[1][t] = inputs[1][t - delay];
        }
    }

    void generateHub()
    {
        inputs.assign(2, vector<double>(params.size_input, 0.0));
        targets.assign(1, vector<double>(params.size_input, 0.0));
        uniform_real_distribution<> dist(0.0, 1.0);

        for (int t = 0; t < params.size_input; t++)
        {
            inputs[0][t] = dist(rng);
            inputs[1][t] = dist(rng);
        }
        int delay = 5;
        for (int t = delay; t < params.size_input; t++)
        {
            targets[0][t] = inputs[0][t - delay] * inputs[1][t - delay]; // Integración no-lineal
        }
    }

    void generateTimescale()
    {
        inputs.assign(1, vector<double>(params.size_input, 0.0));
        targets.assign(1, vector<double>(params.size_input, 0.0));
        uniform_real_distribution<> dist(-1.0, 1.0);

        for (int t = 0; t < params.size_input; t++)
            inputs[0][t] = dist(rng);

        int fast = 1, slow = 15; // Un modo rápido y uno muy lento
        for (int t = slow; t < params.size_input; t++)
        {
            targets[0][t] = inputs[0][t - fast] + inputs[0][t - slow];
        }
    }

    void generateFastProduct()
    {
        inputs.assign(1, vector<double>(params.size_input, 0.0));
        targets.assign(1, vector<double>(params.size_input, 0.0));
        uniform_real_distribution<> dist(-1.0, 1.0);

        for (int t = 0; t < params.size_input; t++)
            inputs[0][t] = dist(rng);

        for (int t = 2; t < params.size_input; t++)
        {
            // Strictly penalizes hysteresis and rewards high-frequency separation
            targets[0][t] = inputs[0][t - 1] * inputs[0][t - 2];
        }
    }

    void generateClassification()
    {
        num_samples = 400;
        seq_length = 40; // Shortened slightly to prevent total noise-washout
        class_sequences.assign(num_samples, vector<double>(seq_length, 0.0));
        class_labels.assign(num_samples, 0.0);

        uniform_real_distribution<> noise_dist(-1.0, 1.0);    // Heavy noise
        uniform_real_distribution<> phase_dist(0.0, 6.28318); // Random phase (0 to 2*PI)

        for (int i = 0; i < num_samples; i++)
        {
            // Strictly balance classes
            int label = (i % 2 == 0) ? 1 : -1;
            class_labels[i] = label;

            // The Random Phase ensures the final time-step is mathematically useless.
            // The connectome MUST use its topology to classify the trajectory.
            double phase = phase_dist(rng);

            for (int t = 0; t < seq_length; t++)
            {
                double signal = 0.0;
                if (label == 1)
                {
                    signal = sin(0.2 * t + phase); // Slow dynamics
                }
                else
                {
                    signal = sin(0.8 * t + phase); // Fast dynamics
                }

                // Signal + Noise forces the network to rely on robust structural modules
                class_sequences[i][t] = signal + noise_dist(rng);
            }
        }
    }

public:
    Task(TaskParams &p) : params(p)
    {
        if (params.type == MULTITASK)
            generateMultitask();
        else if (params.type == HUB)
            generateHub();
        else if (params.type == TIMESCALE)
            generateTimescale();
        else if (params.type == MEMORY_CAPACITY)
            generateMemoryCapacity();
        else if (params.type == NARMA10)
            generateNARMA10();
        else if (params.type == FAST_PRODUCT)
            generateFastProduct();
        else if (params.type == CLASSIFICATION)
            generateClassification();
    }
};

#endif