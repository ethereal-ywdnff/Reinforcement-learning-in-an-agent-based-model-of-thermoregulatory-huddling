import math

import numpy as np


n_agents = 12                                       # number of agents
ambient_temp_set = np.array([10, 13, 16, 19, 22, 25])   # ambient temperature set
alpha = 3.0                                         # fitness weighting
n_sensors = 1000                                    # number of sensors (per agent)
t1 = 1000                                           # number of iterations of huddling dynamics
t0 = 200                                            # start time for averaging
r = 1.0                                             # radius of circular agent
ra = 10.*r                                          # arena radius
Gmax = 5.0                                          # metabolic rates in range 0, Gmax
Kmax = 5.0                                          # thermal (contact) conductance in range 0, Kmax
k1 = 1.0                                            # thermal conductance (rate of heat exchange with environment)
v = 0.3                                             # forward velocity
vr = 200.0                                          # rotation velocity
sigma = -1./100.0                                   # constant for sensor/motor mapping
preferred_temp = 37.0                               # preferred body temperature
dt = 0.05                                           # integration time constant


if __name__ == '__main__':
    print(f"Number of Agents: {n_agents}")

    with open("x_y.txt", "a") as xyfile:

        for d in range(len(ambient_temp_set)):  # simulation under different ambient temperature
            ambient_temp = ambient_temp_set[d]
            print("Starting...", end='')
            print(f"\tAmbient Temperature: {ambient_temp}")

            G = np.random.rand(n_agents) * Gmax
            K2 = np.random.rand(n_agents) * Kmax

            x = np.zeros(n_agents)
            y = np.zeros(n_agents)
            theta = np.zeros(n_agents)

            LR = np.zeros((n_agents, n_sensors), dtype=int)
            tau = np.zeros((n_agents, n_sensors))
            DK = np.zeros((n_agents, n_sensors))

            TL = np.zeros(n_agents)
            TR = np.zeros(n_agents)
            Tc = np.zeros(n_agents)
            Tb = np.zeros(n_agents)
            TbSum = np.zeros(n_agents)
            A = np.zeros(n_agents)

            # Location of thermometers on agent circumference
            xk = np.zeros(n_sensors)
            yk = np.zeros(n_sensors)
            phik = np.zeros(n_sensors)
            for k in range(n_sensors):
                Phi = float(k) * 2.0 * math.pi / float(n_sensors) - math.pi
                phik[k] = Phi
                xk[k] = r * math.cos(Phi)
                yk[k] = r * math.sin(Phi)

            # Simulation constants
            over2pi = 1.0 / (2.0 * math.pi)
            overdn = 1.0 / float(n_sensors)
            overdN = 1.0 / float(n_agents)
            piOver2 = math.pi * 0.5
            r2 = r * r
            r2x4 = 4.0 * r2
            nnorm = 1.0 / (float(n_sensors) * 2.0)
            norm = 1.0 / (float(t1 - t0))
            normOverDt = norm / dt

            # Reset positions and orientations
            for i in range(n_agents):
                theta_init = np.random.rand() * 2.0 * math.pi
                rho_init = np.random.rand() * r
                x[i] = rho_init * math.cos(theta_init)
                y[i] = rho_init * math.sin(theta_init)
                theta[i] = (np.random.rand() - 0.5) * 2. * math.pi
                Tb[i] = preferred_temp
                xyfile.write(f"{x[i]},{y[i]},")


            for t in range(t1):
                # Compute distances between agents
                dx, dy, dkx, dky, dk2 = 0.0, 0.0, 0.0, 0.0, 0.0
                for i in range(n_agents):
                    for k in range(n_sensors):
                        DK[i][k] = 1e9
                        tau[i][k] = ambient_temp
                    for j in range(n_agents):
                        if i != j:
                            dx = x[j] - x[i]
                            dy = y[j] - y[i]
                            if dx * dx + dy * dy <= r2x4:
                                for k in range(n_sensors):
                                    dkx = x[j] - (x[i] + xk[k])
                                    dky = y[j] - (y[i] + yk[k])
                                    dk2 = dkx * dkx + dky * dky
                                    if dk2 < r2 and dk2 < DK[i][k]:
                                        DK[i][k] = dk2
                                        tau[i][k] = Tb[j]

                # Compute contact temperatures and exposed areas
                for i in range(n_agents):
                    Tc[i] = 0.0
                    contact = 0
                    for k in range(n_sensors):
                        if DK[i][k] < 1e9:
                            Tc[i] += tau[i][k]
                            contact += 1
                    if contact:
                        Tc[i] /= float(contact)
                        A[i] = 1.0 - (float(contact) * overdn)
                    else:
                        Tc[i] = 0.0
                        A[i] = 1.0

                # Use theta to assign sensors to Left or Right of body and average
                for i in range(n_agents):
                    TL[i] = 0.0
                    TR[i] = 0.0
                    for k in range(n_sensors):
                        LR[i][k] = int(math.fabs(math.pi - math.fabs(
                            (math.pi - math.fabs((theta[i] + piOver2) % (2.0 * math.pi) - phik[k]))) < piOver2))
                        if LR[i][k]:
                            TL[i] += tau[i][k]
                        else:
                            TR[i] += tau[i][k]
                    TL[i] *= nnorm
                    TR[i] *= nnorm

                # Update body temperatures
                for i in range(n_agents):
                    Tb[i] += (K2[i] * (1.0 - A[i]) * (Tc[i] - Tb[i]) - k1 * A[i] * (Tb[i] - ambient_temp) + G[i]) * dt

                # Rotate and move agents forwards and enforce circular boundary
                for i in range(n_agents):
                    sR = 1.0 / (1.0 + math.exp(sigma * (preferred_temp - Tb[i]) * TR[i]))
                    sL = 1.0 / (1.0 + math.exp(sigma * (preferred_temp - Tb[i]) * TL[i]))

                    theta[i] += math.atan(vr * (sL - sR) / (sL + sR)) * dt

                    x[i] += math.cos(theta[i]) * v * dt
                    y[i] += math.sin(theta[i]) * v * dt

                    rho = math.sqrt(x[i] * x[i] + y[i] * y[i])
                    if (rho + r) >= ra:
                        x[i] += (ra - rho - r) * x[i] / rho * dt
                        y[i] += (ra - rho - r) * y[i] / rho * dt

                    # xyfile.write(f"{x[i]},{y[i]},")

                # Keep contact agents away from each other
                touching = [[False for _ in range(n_agents)] for _ in range(n_agents)]

                vx = np.zeros(n_agents)
                vy = np.zeros(n_agents)
                dx2, dy2, d2, f = 0.0, 0.0, 0.0, 0.0
                for i in range(n_agents):
                    for j in range(n_agents):
                        if i != j:
                            dx2 = x[j] - x[i]
                            dy2 = y[j] - y[i]
                            d2 = dx2 * dx2 + dy2 * dy2
                            if d2 <= r2x4:
                                f = min((r - math.sqrt(d2) * 0.5), r) / math.sqrt(d2)
                                vx[j] += f * dx2
                                vy[j] += f * dy2
                                touching[i][j] = True
                for i in range(n_agents):
                    x[i] += vx[i] * dt
                    y[i] += vy[i] * dt
                    xyfile.write(f"{x[i]},{y[i]},")



