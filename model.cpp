/*
g++ model.cpp -o model
./model output.txt 0
./model learning.txt 1
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
void filialHuddling(int n);

int main(int argc, char** argv){
    
    // seed random number generator using current clock time (i.e., a different simulation each time)
    time_t seed = time(NULL);
    std::cout << "Seed number: " << seed << std::endl;
    srand(seed);
    // srand(10);
    
    
    // Supplied model parameters
    int N = 12;                   // number of agents
    double Ta = 20;               // ambient temperature
    double alpha = 3.0;           // fitness weighting
    
    // logfile
    std::stringstream ss;
    ss<<argv[1];
    ofstream logfile;
    logfile.open(ss.str().c_str(),ios::out|ios::trunc);

    // position file
    std::stringstream ss1;
    ss1<<"position.txt";
    ofstream position;
    position.open(ss1.str().c_str(),ios::out|ios::trunc);

    // association file
    std::stringstream ss2;
    ss2<<"association.txt";
    ofstream association;
    association.open(ss2.str().c_str(),ios::out|ios::trunc);


    
    // Preset model parameters
    int n = 1000;                           // number of sensors (per agent)
    int t1 = 1000;                          // number of iterations of huddling dynamics (within generation; NOTE: set to 10000 if reproducing Figure 5)
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
    double gamma = 0.001;
    double leanring = stod(argv[2]);
    // double p = 0.0;
    
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
    
    for (int o=0; o<5; o++) {
        for (int day=0; day<60;day+=5){
            Tp = (40-32)*exp(-day/10)+32;
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
                    
                    if (leanring == 1){
                        theta[i] += atan(Vr*(sL-sR)/(sL+sR))*dt*p[i]*50;
                    } else {
                        theta[i] += atan(Vr*(sL-sR)/(sL+sR))*dt;
                    }
                    // V += V*p[i];
                    
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
                    // learning
                    reward = abs(Tb[i] - Tp) > abs(tb_prev[i] - Tp);
                    p[i] = 0.0;
                    for (int j = 0; j < N; j++) {
                        p[i] += w[i][j] * touching[i][j];
                    }
                    
                    for (int j = 0; j < N; j++) {
                        w[i][j] += gamma * (reward - p[i]) * touching[i][j];
                        // cout << w[i][j] << endl;
                        association<<w[i][j]<<",";
                    }
                }
                // for(int i=0;i<N;i++){
                //     association<<p[i]<<",";
                // }
                
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
            }
            logfile<<huddling<<","<<Ta<<",";

        }
    }
    logfile.close();
    position.close();
    association.close();


    // system("python vis.py");

    // filialHuddling(N);

    return 0;
};

void filialHuddling(int n) {
    srand(1);
    int timesteps = 50000;
    double dt = 60./(double)timesteps;
    double gamma = 0.001;   // Delta-rule learning-rate
    double beta = 1./5.;
    
    // Q maintains associative strengths (alpha)
    vector<vector<double> > Q(n);
    for(int i=0;i<n;i++){
        if(gamma == 0.){
            Q[i].resize(n,1.);
        } else {
            Q[i].resize(n,0.);
        }
        Q[i][i] = 0.;
    }
    
    // logfile
    std::stringstream ss;
    ss<<"learning.txt";
    ofstream learning;
    learning.open(ss.str().c_str(),ios::out|ios::trunc);
    learning<<n<<",";    // record N first
    
    vector<int> I(n,0); // ID of the group it belongs to
    vector<int> S(n,0);
    
    // constants
    double k = 8.31;
    double c1 = 19.;
    double c2 = 3.;
    double c3 = 1./40.;
    // double Tp = 35.7;
    
    double TIME = 0.;
    for(int t=0;t<timesteps;t++){
        
        TIME += dt;
        
        bool monte = true;
        
        for (int i=0;i<n;i++){
            S[i] = 0;
        }
        
        for (int i=0;i<n;i++){
            S[I[i]]++;
        }
        
        // model
        double P  = exp(-TIME/k);       // brown fat depletion
        double G  = k*(1.-k*P*log(P)); // metabolic rate as entropy
        double M  = k*(1.-P);          // muscle mass
        double T1 = c2*P;              // temperature preference 1
        double N  = c1*exp(-k*T1);     // non-muscle mass
        double T2 = c3*N*G;            // temperature preference 2
        // cout <<T1<<" "<<T2<< endl;
        
        // number of groups
        vector<int> groups(0);
        int Ngroups = 0;
        for(int i=0;i<n;i++){
            if(S[i]>0){
                groups.push_back(i);
                Ngroups++;
            }
        }
        
        // randomly picked individual
        int a = floor(randDouble()*n);
        
        double reward = 0.;
        
        if (Ngroups==1){
            I[a]=(I[a]+1)%n;
        }
        else if (Ngroups==n){
            int b=a;
            while(I[b]==I[a]){
                b = floor(randDouble()*n);
            }
            I[b]=I[a];
        }
        else {
            int b=a;
            while(I[b]==I[a]){
                b = floor(randDouble()*n);
            }
            
            // Thermodynamic temperature
            double T = 1./(1.+exp(-Q[b][a]*(T1-T2)*beta));
            
            
            // Join together the groups to which a and b belong
            if (randDouble()<T){
                
                reward = 1.;
                vector<int> A(0);
                for(int i=0;i<n;i++){
                    if(I[i]==I[a]){
                        A.push_back(i);
                    }
                }
                for(int i=0;i<A.size();i++){
                    I[A[i]] = I[b];
                }
            }
            
            // Split pup a from its group
            else {
                reward = 0.;
                vector<int> Q(n);
                for(int i=0;i<Ngroups;i++){
                    Q[groups[i]] = 1;
                }
                for(int i=0;i<n;i++){
                    if(Q[i]==0){
                        I[a] = i;
                        break;
                    }
                }
            }

            // Update associative strengths
            double sumQ = 0.;
            for(int j=0; j<n;j++){
                if(!(j==a)){
                    sumQ += Q[a][j];
                }
            }
            Q[a][b] += gamma*(reward-sumQ);
        }
        
        // Log data to text file
        for(int i=0;i<n;i++){
            learning<<S[I[i]]<<","<<Q[0][i]<<","<<Q[i][0]<<",";
        }
        learning<<t*dt<<",";
        learning<<endl;
        
    }
    
    learning.close();
}

double randDouble(void){
    /* 
     Returns a random double from the uniform distribution in the range 0 to 1
     */
    return ((double) rand())/(double)(RAND_MAX);
}




