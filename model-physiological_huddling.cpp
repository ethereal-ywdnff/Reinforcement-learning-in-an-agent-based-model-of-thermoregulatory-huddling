/*
g++ model-physiological_huddling.cpp -o physiological_huddling
./physiological_huddling

This model is to simulate the physiological huddling, which can generate data used to visualize the movements and plot graphs 
of body temperature and huddling under different ambient temperatures.
*/


#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
using namespace std;

double randDouble(void);

int main(int argc, char** argv){
    
    // seed random number generator using current clock time (i.e., a different simulation each time)
    time_t seed = time(NULL);
    std::cout << "Seed number: " << seed << std::endl;
    srand(seed);
    // srand(10);
    
    
    // Supplied model parameters
    int N = 12;                   // number of agents
    double Ta = 0;                // ambient temperature
    vector<double> Taa(36,0.);    // several different ambient temperatures (from 10 to 35)
    for (int i=0; i<36; i++){
        Taa[i] = Ta;
        Ta+=1;
    }

    
    double alpha = 3.0;           // fitness weighting
    
    // logfile
    std::stringstream ss;
    ss<<"output1.txt";
    ofstream logfile;
    logfile.open(ss.str().c_str(),ios::out|ios::trunc);

    // position file
    std::stringstream ss1;
    ss1<<"position1.txt";
    ofstream position;
    position.open(ss1.str().c_str(),ios::out|ios::trunc);


    // Preset model parameters
    int n = 1000;                           // number of sensors (per agent)
    int t1 = 1000;                          // number of iterations of huddling dynamics
    int t0 = 200;                           // start time for averaging (within generation)
    double r = 1.0;                         // radius of circular agent
    double ra = 10.*r;                      // arena radius
    double Gmax = 5.0;                      // metabolic rates in range 0, Gmax
    double Kmax = 5.0;                      // thermal (contact) conductances in range 0, Kmax
    double k1 = 1.0;                        // thermal conductance (rate of heat exchange with environment)
    double V = 0.3;                         // forward velocity
    double Vr = 200.;                       // rotation velocity
    double sigma = -1./100.;                // constant for sensor/motor mapping
    double Tp = 37.;                        // preferred body temperature
    double dt = 0.05;                       // integration time constant
    
    // Evolvable thermal variables
    vector<double> G (N, Gmax);
    vector<double> K2 (N, Kmax);
    
    // Agent-based geometric variables
    vector <double> x(N,0.);                // x location for N agents
    vector <double> y(N,0.);                // y location for N agents
    vector <double> theta(N,0.);            // orientation for N agents
    vector<double> p(N, 0.0);
    vector<vector<double> > w(N, vector<double>(N, 0.));
    vector<double> tb_prev(N, 0.);
    double reward = 0.;
    vector <vector <int> > LR (N);          // label sensors as on the left (1) or right (0)
    vector <vector <double> > tau (N);      // temperature sensors
    vector <vector <double> > DK (N);       // distance of taxel from nearest pup
    
    for(int i=0;i<N;i++){
        LR[i].resize(n);
        tau[i].resize(n);
        DK[i].resize(n);
    }
    
    // Agent-based thermal variables
    vector<double> TL (N);
    vector<double> TR (N);
    vector<double> Tc (N);
    vector<double> Tb (N);
    vector<double> TbSum (N);
    vector<double> A(N);
    
    // Location of thermometers on agent circumference
    vector <double> xk(n);
    vector <double> yk(n);
    vector <double> phik(n);
    for(int k=0;k<n;k++){
        double Phi = (double)k*2.*M_PI/(double)n-M_PI;
        phik[k] = Phi;
        xk[k] = r*cos(Phi);
        yk[k] = r*sin(Phi);
    }
    
    // Misc simulation constants (pre-computed for speed)
    double over2pi = 1.0/(2.*M_PI);
    double overdn = 1.0/double(n);
    double overdN = 1.0/double(N);
    double piOver2 = M_PI*0.5;
    double r2 = r*r;
    double r2x4 = 4.*r2;
    double nnorm = 1.0/(double(n)*2.);
    double norm = 1.0/((double)(t1-t0));
    double normOverDt = norm/dt;
    
    
    
    /*
     START OF MAIN SIMUALTION
     */
    
    for (int o=0; o<Taa.size(); o++) {
        Ta = Taa[o];
            
        // Reset positions and orientations
        for (int i=0;i<N;i++){
            double theta_init = randDouble()*M_PI*2.;
            // double rho_init = randDouble()*r;
            double rho_init = 0;
            x[i] = rho_init*cos(theta_init);
            y[i] = rho_init*sin(theta_init);
            theta[i] = (randDouble()-0.5)*2.*M_PI;
            Tb[i] = Tp;
            position<<x[i]<<","<<y[i]<<","<<Tb[i]<<",";
            TbSum[i] = 0.;
        }
        // Reset metrics
        double huddling = 1.;
        double groups = 0.;
        vector<double> Adifference(N,0.);
        vector<double> Aprevious(N,0.);

        // INNER LOOP ITERATES HUDDLING TIMESTEPS
        for(int t=0;t<t1;t++){
            
            // Compute distances between agents; where overlapping set sensor to T_B of contacting agent
            double dOver2r, phi, dx, dy, dkx, dky, dk2;
            for(int i=0;i<N;i++){
                for(int k=0;k<n;k++){
                    DK[i][k] = 1e9;
                    tau[i][k] = Ta;
                }
                for(int j=0;j<N;j++){
                    if(i!=j){
                        dx = x[j]-x[i];
                        dy = y[j]-y[i];
                        if(dx*dx+dy*dy<=r2x4){
                            for(int k=0;k<n;k++){
                                dkx = x[j]-(x[i]+xk[k]);
                                dky = y[j]-(y[i]+yk[k]);
                                dk2 = dkx*dkx+dky*dky;
                                if(dk2 < r2 && dk2 < DK[i][k]){
                                    DK[i][k]=dk2;
                                    tau[i][k]=Tb[j];
                                }
                            }
                        }
                    }
                }
            }
            
            // Compute contact temperatures T_C and exposed areas A
            for(int i=0;i<N;i++){
                Tc[i]=0.;
                int contact = 0;
                for(int k=0;k<n;k++){
                    if(DK[i][k] < 1e9){
                        Tc[i] += tau[i][k];
                        contact++;
                    }
                }
                if(contact){
                    Tc[i] /= (double)contact;
                    A[i] = 1.-((double)contact*overdn);
                } else {
                    Tc[i] = 0.;
                    A[i] = 1.;
                }
            }
            
            // Use theta to assign sensors to Left or Right of body and average
            for(int i=0;i<N;i++){
                TL[i]=0.;
                TR[i]=0.;
                for(int k=0;k<n;k++){
                    LR[i][k]=(int)(M_PI-fabs(M_PI-fabs(fmod(theta[i]+piOver2,2.*M_PI)-phik[k]))<piOver2);
                    if(LR[i][k]){
                        TL[i] += tau[i][k];
                    } else {
                        TR[i] += tau[i][k];
                    }
                }
                TL[i] *= nnorm;
                TR[i] *= nnorm;
            }
            
            
            // Update body temperatures
            for(int i=0;i<N;i++){
                tb_prev[i] = Tb[i];
                Tb[i] += (K2[i]*(1.-A[i])*(Tc[i]-Tb[i])-k1*A[i]*(Tb[i]-Ta)+G[i])*dt;
            }
            
            // Rotate and move agents forwards and enforce circular boundary
            for(int i=0;i<N;i++){
                
                double sR = 1./(1.+exp(sigma*(Tp-Tb[i])*TR[i]));
                double sL = 1./(1.+exp(sigma*(Tp-Tb[i])*TL[i]));
                
                theta[i] += atan(Vr*(sL-sR)/(sL+sR))*dt;

                x[i] += cos(theta[i])*V*dt;
                y[i] += sin(theta[i])*V*dt;
                
                double rho = sqrt(x[i]*x[i]+y[i]*y[i]);
                if((rho+r)>=ra){
                    x[i] += (ra-rho-r)*x[i]/rho*dt;
                    y[i] += (ra-rho-r)*y[i]/rho*dt;
                }
            }
            
            // spring contacting agents away from each other
            
            vector<vector<bool> > touching(N);
            for(int k=0;k<N;k++){
                touching[k].resize(N);
            }
            vector<double> vx(N,0.);
            vector<double> vy(N,0.);
            double dx2, dy2, d2, f;
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    if(i!=j){
                        dx2 = x[j]-x[i];
                        dy2 = y[j]-y[i];
                        d2 = dx2*dx2+dy2*dy2;
                        if((d2<=r2x4)){
                            f = fmin(r-sqrt(d2)*0.5,r)/sqrt(d2);
                            vx[j] += f*dx2;
                            vy[j] += f*dy2;
                            touching[i][j]=true;
                        }
                    }
                }
            }
            for(int i=0;i<N;i++){
                x[i] += vx[i]*dt;
                y[i] += vy[i]*dt;
                position<<x[i]<<","<<y[i]<<","<<Tb[i]<<",";
            }
            
            // increment huddling metrics
            if(t>=t0){
                for(int i=0;i<N;i++){
                    TbSum[i] += Tb[i];
                }
                double hud = 0.;
                for(int i=0;i<N;i++){
                    hud += 1.-A[i];
                }
                hud *= overdN;
                for(int i=0;i<N;i++){
                    Adifference[i] += fabs(A[i]-Aprevious[i]);
                    Aprevious[i] = A[i];
                }
                huddling += hud;
            }
        }
        huddling *= norm;
        groups *= norm;

        // identify least fit agent and log metrics
        for(int i=0;i<N;i++){
            double TbAvg = TbSum[i]*norm;

            logfile<<TbAvg<<",";
            // cout<<TbAvg<<endl;
        }
        logfile<<huddling<<","<<Ta<<",";


    }
    logfile.close();
    position.close();

    // system("python vis.py");

    return 0;
};


double randDouble(void){
    /* 
     Returns a random double from the uniform distribution in the range 0 to 1
     */
    return ((double) rand())/(double)(RAND_MAX);
}