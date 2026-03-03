#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<format>
#include "ESN.h"

using namespace std;

vector<vector<double>> read_matrix(std::ifstream &infile);

int main(){

ESNParams params;

params.reservoir_size=20;   //number of regions in the atlas
params.spectral_radius=0.9;  
params.sparsesity=0.1;
params.connectivity_reservoir=10;
params.input_size= 1;
params.output_size=10;
params.tau_vals=10; // Time lags for memory capacity. We can use a vector if we want to test different taus.


// 1) Connectome with high disparity
std::ifstream infile("connectome_high_disparity.dat");
std::vector<std::vector<double>> connectome_hdis = read_matrix(infile);

ESN esn_hdis(params);
esn_hdis.setW(connectome_hdis);
cout << format("Disparity HDL: {:.4f}", esn_hdis.calculateDisparity(connectome_hdis)) << endl;

// 2) Connectome with low disparity
std::ifstream infile2("connectome_low_disparity.dat");
std::vector<std::vector<double>> connectome_ldis = read_matrix(infile2);
ESN esn_ldis(params);
esn_ldis.setW(connectome_ldis);
cout << format("Disparity LDL: {:.4f}", esn_ldis.calculateDisparity(connectome_ldis)) << endl;

// 3) Randomly generated reservoir

ESN esn(params);
cout<<format("\nRandomly generated reservoir")<<endl;
esn.printSpectralRadius();
esn.printDisparity();
esn.rescaleReservoir(params.spectral_radius);
esn.printSpectralRadius();

//esn.EvoluteReservoir();

return 0;

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