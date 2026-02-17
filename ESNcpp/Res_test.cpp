#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<format>
#include "ESN.h"

using namespace std;

vector<vector<double>> read_matrix(std::ifstream &infile);
double Disparity(const vector<vector<double>> &matrix);

int main(){

ESNParams params;

params.reservoir_size=20;   //number of regions in the atlas
params.spectral_radius=0.9;  
params.sparsesity=0.1;
params.connectivity_reservoir=10;
params.input_size= 1; //depende de la tarea. Si fuera MNITS, seria 28^2=784. Podemos tomar 1 como ejemplo. 
//proyeccion aleatoria las regiones en nuestro caso que solo tenemos conectomas. 
//en el caso de LE Suarez, serían las regiones subcorticales. 
// en el de de damicelli, hacen lo que nosotros (proyectar aleatorios entre -1 y 1)
params.output_size=10; //arbitrario en nuestro caso. depende de la tarea. Vamos a hacer como si fueramos a hacer MNIST

// 1) Connectome with high disparity
std::ifstream infile("connectome_high_disparity.dat");
std::vector<std::vector<double>> connectome_hdis = read_matrix(infile);

ESN esn_hdis(params);
esn_hdis.setW(connectome_hdis);
cout << std::format("Disparity HDL: {:.4f}", esn_hdis.calculateDisparity(connectome_hdis)) << endl;

// 2) Connectome with low disparity
std::ifstream infile2("connectome_low_disparity.dat");
std::vector<std::vector<double>> connectome_ldis = read_matrix(infile2);
ESN esn_ldis(params);
esn_ldis.setW(connectome_ldis);
cout << std::format("Disparity LDL: {:.4f}", esn_ldis.calculateDisparity(connectome_ldis)) << endl;

// 3) Randomly generated reservoir

ESN esn(params);

esn.printSpectralRadius();
esn.printDisparity();
esn.rescaleReservoir(params.spectral_radius);
esn.printSpectralRadius();
esn.EvoluteReservoir();



//I can rescale all to have spectral radius 0.99, and then compare the dynamics of the two reservoirs.



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