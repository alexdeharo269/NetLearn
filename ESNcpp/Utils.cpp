
#include<vector>
#include<fstream>
#include<sstream>

using namespace std;

vector<vector<double>> read_matrix(std::ifstream &infile)
{
    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(infile, line))
    {
        // Ignorar líneas vacías
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::vector<double> row;
        double value;

        // El operador >> salta automáticamente espacios y tabuladores
        while (iss >> value)
        {
            row.push_back(value);
        }

        if (!row.empty())
        {
            matrix.push_back(row);
        }
    }
    return matrix;
}

// ... mantén process_csv exactamente igual que antes ...
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

        if (row_vals.size() < (size * size) + 1)
            continue;

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

// Utilidad para extraer hemisferios en el Multitask
vector<vector<double>> mask_states(const vector<vector<double>> &full_states, int start_idx, int end_idx)
{
    vector<vector<double>> masked(full_states.size(), vector<double>(end_idx - start_idx, 0.0));
    for (size_t t = 0; t < full_states.size(); ++t)
    {
        for (int n = start_idx; n < end_idx; ++n)
        {
            masked[t][n - start_idx] = full_states[t][n];
        }
    }
    return masked;
}