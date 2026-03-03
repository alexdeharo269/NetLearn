
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