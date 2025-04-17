# 🔄 Tunnel Network Simulation (Python + SimPy + NetworkX)

This project simulates a dynamic, queue-based packet network in a tunnel-like topology. It models real-world constraints such as:
- Link latency and bandwidth
- Queue size limitations
- Packet drops due to buffer overflow or unreliable links
- Dynamic traffic generation
- Animated visualization of packet flow - for easier understanding

## 🧠 Technologies Used
- **SimPy**: for event-driven packet scheduling and simulation time control
- **NetworkX**: for building and managing the network graph
- **Matplotlib**: for visualizing the topology and animated packet flows
- **Pandas**: for logging and analyzing simulation output

---

## 🚀 Features

### ✅ Core Simulation:
- Packet transmission with latency and bandwidth constraints
- Configurable queue size (per link)
- Random link failures and probabilistic packet drops
- Multiple concurrent traffic generators
- Detailed per-device statistics and packet logs

### 🎥 Visualization:
- **Real-time packet animation** with matplotlib
- **Topology viewer** with throughput and latency metrics (optional)
- Delay distribution and per-device throughput plots

---

## 🧪 How to Run

### 1. Install dependencies:
```bash
pip install simpy networkx matplotlib pandas numpy
2. Run the simulation:
You can run the simulation with:

python main.py
This will launch a real-time animation of packet flow across your simulated tunnel network.

3. Optional: Enable other plots
Uncomment any of the following lines at the bottom of main.py to enable static visualizations:

plot_topology(graph)
plot_delay_distribution(packet_log)
plot_throughput(stats_df)
unified_topology_plot(graph, stats_df)
```
# ⚙️ **Parameters You Can Tweak**

Inside run_simulation():

num_segments: number of tunnel segments
devices_per_segment: how many nodes per segment
sim_time: how long to run the simulation
rate: average packet generation rate per traffic generator
queue_size: max number of packets per link buffer

# 💡 Future Extensions:
TCP support (ACKs, retransmission, congestion control)
Node mobility and dynamic topology rewiring
Real-time traffic injection or visual dashboards
Failure injection, recovery, and repair logic

# 📁 Project Structure

File	Description
main.py	Core simulation and visualization logic
README.md	This file
requirements.txt (optional)	Add for easy pip install

# 📸 Preview

<img src="https://github.com/user-attachments/assets/7db1d959-7c39-466f-aa0c-04ac5ff8a8c4" width="600"/>


# 📄 License

MIT License © 2024 Yotam Tsabari

---
