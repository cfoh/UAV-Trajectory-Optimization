# UAV Trajectory Optimization

The aim of this project is to repeat the experiments given in the following paper and further enhance the algorithms for improved performance.

- H. Bayerlein, P. De Kerret and D. Gesbert, "Trajectory Optimization for 
  Autonomous Flying Base Station via Reinforcement Learning," IEEE 19th 
  International Workshop on Signal Processing Advances in Wireless 
  Communications (SPAWC), 2018, pp. 1-5.

The UAV scenario given the paper is shown below.

<img src="https://github.com/user-attachments/assets/f63f2188-869c-45a6-bce3-106838bc4ad6" width="400" height="450">

## The Settings

The settings of the scenario are:
- A 15-by-15 grid map.
- The map contains some obstacles (shown in dark grey).
- The UAV makes a trip of 50 steps from the start cell to the final landing cell.
- The start cell is the left-bottom cell (colored in green), 
  and the final landing cell is also the left-bottom cell (marked by `X`).
- The UAV can only move up/down/left/right in each time step from the center of
  one cell to the center of another cell. It cannot move out of the map 
  nor into the obstacle cells.
- There are two stationary users (or called UEs here) on the map. 
  Their locations are marked as blue circles.
- The UAV communicates simultaneously to the two UEs. Due to the obstacles,
  signals at some cells experience non-line-of-sight (NLOS).
  - Clear cells indicate line-of-sight (LOS) communications with both UEs
  - Light grey cells indicate NLOS from a UE
  - Darker grey cells indicate NLOS from both UEs
  - Dark cells indicate obstacles
- The screenshot shows that the UAV (i.e. the red circle) 
  has just completed 49 time steps of its flight time.

## The Objective

The objective of the design is to propose a learning algorithm such that the UAV makes a 50-step trip from the start cell to the final landing cell while providing good communication service to both UEs. 
The paper has proposed two machine learning (ML) algorithms, namely Q-learning and Deep-Q-Network (DQN), to learn the optimal trajectory.
After sufficient training on the algorithms, the authors observe that:
- the UAV is able to discover an optimal region to serve both UEs
- the UAV reaches the optimal region on a short and efficient path
- the UAV avoids flying through the shadowed areas as the communication
  quality in those areas is not excellent
- after reaching the optimal region, the UAV circles around the 
  region to continue to provide good communication service
- the UAV decides when to start returning back to avoid crashing

## The Code

The code is tested with `Python 3.9.20` with the following 
packages (i.e. pygame, shapely, torch):
```
Package                  Version
------------------------ ----------
filelock                 3.16.1
fsspec                   2024.9.0
Jinja2                   3.1.4
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.2.1
numpy                    2.0.2
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.6.68
nvidia-nvtx-cu12         12.1.105
pip                      23.0.1
pygame                   2.6.0
setuptools               58.1.0
shapely                  2.0.6
sympy                    1.13.3
torch                    2.4.1
triton                   3.0.0
typing_extensions        4.12.2
```

Run the simulation by:
```bash
python uav.py
```

By default, the simulation will run Q-Learning algorithm in stateless mode with animation. The folder also contains the following saved models. To see the UAV trajectory produced by the policy of a saved model, simply run the simulation in testing mode.

- Q-Learning model `Q-Learning-load.json` that has been trained for over 800000 episodes using Q-Learning algorithm.

### Scenario Parameters

As the paper does not provide full detail of their scenario setup, we make the following assumptions:
- the map is set to 1600-by-1600 in meters
- the UAV is flying at 40 meters above ground
- For the communication, the thermal noise is -174 dBm per Hz

With the above settings, the rate per UE is in the range between around 35 bps/Hz and 50 bps/Hz which is much higher than that of the paper as illustrated in Fig. 3 in the paper. The paper may have used a much wider area than our considered area, and/or the UAV is flying at a much higher altitude.

### Reward Setting

The paper uses transmission rate as the reward for the learning. Depending on the setup, one can use the rate of downlink or uplink transmissions, or both. For uplink transmission, if the transmissions are not orthogonal (in time or frequency), then transmissions of the UEs will interfere with each other. It is unclear which option is used in the paper, here we use orthogonal transmissions and the transmission can be either uplink or downlink. Without loss of generality, we consider uplink transmissions where in our scenario, there are two IoT devices and a UAV. The mission of the UAV is to collect the data from the IoT devices.

Besides, the paper pointed out that the optimal region to serve both UEs is around the shortest midpoint between the two UEs. However, using sum rate as the reward as indicated in (6) in the paper will not create an optimal region at around the shortest midpoint between the two UEs, instead the optimal regions will be around each UE. To match the optimal trajectory shown in the paper, we use minimum rate of both which creates the optimal region around the shortest midpoint between the two UEs. That is:
```python
# reward = r1 + r2   # sum rate, optimal region at around either UE1 or UE2
reward = min(r1,r2)  # min rate, optimal region at the midpoint of UE1 and UE2
```

Apart from using the rate as the reward, we also add additional rewards so that the UAV will return to the final landing cell at the end of its trip. We apply the following rewards:
- If the UAV returns to the final landing cell before the end of its trip,
  we apply penalty to inform the UAV of its premature returning. The penalty is the last 
  immediate reward times the unused time steps. That is, the earlier the UAV returns, the
  more penalty it will receive, so that it will learn to avoid returning earlier. The paper did not apply this penalty, but we found it useful.
- If the UAV fails to return to the final landing cell at the end of its trip, 
  we apply penalty which is the immediate reward times 10. This way, the UAV will learn to return to the final landing cell at the end of its trip to avoid the penalty. This penalty is also described in the paper, although what penalty to apply is not mentioned.

Note that the paper also applies penalty when the UAV moves outside of the map. However, in our design, we simply do not allow the UAV to move outside of the map.

## The Results

We show the results of the ML performance below. Similar to the observation of the authors, the Q-learning algorithm converged slowly and reached the optimal trajectory at around 800000 rounds (or episodes). 

We also included the results of DQN. As can be seen, DQN converges significantly faster which is indicated by the paper and confirmed by our experiments.

<table>
  <tr>
    <th>Learning convergence<br>
        (Q-learning and DQN)</th>
    <th>Illustration of optimal trajectory with Q-learning<br>
        (reward = 2146.56 after 800000 rounds)</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/013c8a37-6764-443a-a72d-fe7ffaca4a79" width="404">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/12904369-0865-406c-970e-d165c962fc4f" width="402">
    </td>
  </tr>
</table>

Q-learning improves very slowly in the last 100000 rounds:
```
round   reward
======  =======
 ...      ...
630000  2144.75
640000  2145.42
650000  2145.42
660000  2145.44
670000  2145.70
680000  2145.70
690000  2145.70
700000  2145.70
710000  2145.70
720000  2145.92
730000  2145.92
740000  2145.92
750000  2145.90
760000  2146.04
770000  2146.32
780000  2146.32
790000  2146.32
800000  2146.32
810000  2146.56
820000  2146.56
 ...      ...
```
