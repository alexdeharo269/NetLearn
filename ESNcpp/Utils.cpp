
#include<vector>
#include<fstream>
#include<sstream>
#include<tuple>
#include<cmath>
#include<string>

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// readConnectomesWithID — robust I/O contract shared with the notebook.
//
// Fixes a latent mismatch in the previous pipeline: the Python side exported
//   [subject_id, w(0,0), w(0,1), ..., w(N-1,N-1)]   (subject_id FIRST, NO age column)
// while process_csv() below expected the *age* in the LAST column and treated the
// first value as a weight. That silently shifted every matrix by one cell and put a
// weight where the id should be. Here we read the id explicitly from column 0 and the
// flattened N×N matrix (row-major) from the remaining columns, so merges on
// subject_id in pandas are always correct.
//
// Output CSVs written by the runners follow the mirror contract:
//   [subject_id, <metric columns...>]   with a header line.
// ─────────────────────────────────────────────────────────────────────────────
tuple<vector<int>, vector<vector<vector<double>>>>
readConnectomesWithID(std::ifstream &infile, int size, bool zero_diagonal = true)
{
    vector<int> ids;
    vector<vector<vector<double>>> matrices;
    string line;
    bool first_line = true;

    while (getline(infile, line))
    {
        if (line.empty())
            continue;
        if (first_line)         // skip header
        {
            first_line = false;
            continue;
        }

        stringstream ss(line);
        string value;
        vector<double> vals;
        while (getline(ss, value, ','))
            if (!value.empty())
                vals.push_back(stod(value));

        // need id + N*N values
        if ((int)vals.size() < size * size + 1)
            continue;

        ids.push_back((int)llround(vals[0]));

        vector<vector<double>> matrix(size, vector<double>(size, 0.0));
        int flat_index = 1;     // 0 is the subject_id
        for (int r = 0; r < size; ++r)
            for (int c = 0; c < size; ++c)
                matrix[r][c] = vals[flat_index++];

        if (zero_diagonal)      // no self-loops in the reservoir (consistent with the structural analysis)
            for (int i = 0; i < size; ++i)
                matrix[i][i] = 0.0;

        matrices.push_back(std::move(matrix));
    }
    return make_tuple(ids, matrices);
}


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