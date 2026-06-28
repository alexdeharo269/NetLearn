// Two-population E/I rate network:
//   (1) ISN steady state, (2) paradoxical effect, (3) self-balancing inhibition.
// Just the dynamics, forward Euler. Standard library only.
//
//   compile:  g++ -O2 -o isn isn.cpp
//   run:      ./isn        (writes sweep.csv and plastic.csv)

#include <cstdio>

double relu(double x) { return x > 0.0 ? x : 0.0; }

// Integrate the 2D rate model to steady state. Result returned in rE, rI.
void steady_state(double WEE, double WEI, double WIE, double WII,
                  double IE, double II, double &rE, double &rI)
{
    const double tauE = 20.0, tauI = 10.0, dt = 0.1;   // ms
    rE = 0.0; rI = 0.0;
    for (int t = 0; t < 100000; t++) {                 // 10 s >> tau: converged
        double drE = (-rE + relu(WEE*rE - WEI*rI + IE)) / tauE;
        double drI = (-rI + relu(WIE*rE - WII*rI + II)) / tauI;
        rE += dt * drE;
        rI += dt * drI;
    }
}

int main()
{
    // baseline ISN parameters
    double WEE = 1.25, WEI = 1.2, WIE = 1.0, WII = 0.8;
    double IE = 2.0, II = 1.0;
    double rE, rI;

    // ---- Part 1: steady state of the ISN ----
    steady_state(WEE, WEI, WIE, WII, IE, II, rE, rI);
    printf("Part 1  ISN steady state:  rE = %.4f   rI = %.4f\n", rE, rI);
    double rho0 = rE;                       // target rate used in Part 3

    // ---- Part 2: paradoxical effect: sweep the input to I ----
    // ISN vs. a non-ISN control (WEE = 0.6).  ->  sweep.csv
    FILE *f2 = fopen("sweep.csv", "w");
    fprintf(f2, "II,rE_isn,rI_isn,rE_ctrl,rI_ctrl\n");
    for (double Iinj = 0.0; Iinj <= 4.0001; Iinj += 0.1) {
        double eE, eI, cE, cI;
        steady_state(WEE, WEI, WIE, WII, IE, Iinj, eE, eI);   // ISN
        steady_state(0.60, WEI, WIE, WII, IE, Iinj, cE, cI);  // control
        fprintf(f2, "%.2f,%.4f,%.4f,%.4f,%.4f\n", Iinj, eE, eI, cE, cI);
    }
    fclose(f2);
    printf("Part 2  wrote sweep.csv   (rI_isn DECREASES, rI_ctrl INCREASES)\n");

    // ---- Part 3: self-balancing inhibition (plastic WEI) ----
    // Slow inhibitory plasticity:  tau_w dWEI/dt = eta * rI * (rE - rho0).
    // After a step increase in IE at t = 2 s, WEI adapts until rE -> rho0. -> plastic.csv
    const double tauE = 20.0, tauI = 10.0, dt = 0.1;   // ms
    const double k = 0.001;                            // plasticity rate (slow)
    double E = rho0, I = rI;                           // start at Part 1 fixed point
    double w = WEI;                                    // WEI now evolves
    FILE *f3 = fopen("plastic.csv", "w");
    fprintf(f3, "t_ms,rE,rI,WEI,IE\n");
    for (int t = 0; t < 80000; t++) {                  // 8 s total
        double time  = t * dt;
        double drive = (time >= 2000.0) ? IE + 1.5 : IE;     // step IE up at t = 2 s
        double drE = (-E + relu(WEE*E - w*I + drive)) / tauE;
        double drI = (-I + relu(WIE*E - WII*I + II)) / tauI;
        double dw  = k * I * (E - rho0);
        E += dt * drE;
        I += dt * drI;
        w += dt * dw;
        if (w < 0.0) w = 0.0;
        if (t % 50 == 0)                               // record every 5 ms
            fprintf(f3, "%.1f,%.4f,%.4f,%.4f,%.2f\n", time, E, I, w, drive);
    }
    fclose(f3);
    printf("Part 3  wrote plastic.csv (rE returns to rho0 = %.4f as WEI adapts)\n", rho0);

    return 0;
}
