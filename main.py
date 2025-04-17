import networkx as nx
import simpy
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

# Enhanced Packet class with ID
class Packet:
    _id_counter = 0
    def __init__(self, src, dst, timestamp, size=1.0):
        self.id = Packet._id_counter
        Packet._id_counter += 1
        self.src = src
        self.dst = dst
        self.timestamp = timestamp
        self.size = size

# Custom tunnel topology with link status
def create_tunnel_topology(num_segments=5, devices_per_segment=2, failure_rate=0.1):
    G = nx.Graph()
    for seg in range(num_segments):
        for dev in range(devices_per_segment):
            node_id = f"S{seg}_D{dev}"
            G.add_node(node_id, segment=seg)
            for d in range(dev):
                G.add_edge(
                    node_id,
                    f"S{seg}_D{d}",
                    latency=random.uniform(0.2, 2),
                    bandwidth=random.uniform(5, 15),
                    active=True,
                    drop_rate=random.uniform(0.0, 0.1)
                )
        if seg < num_segments - 1:
            G.add_edge(
                f"S{seg}_D0",
                f"S{seg+1}_D0",
                latency=random.uniform(0.4, 1),
                bandwidth=random.uniform(3, 10),
                active=random.random() > failure_rate,
                drop_rate=random.uniform(0.05, 0.1)
            )
    return G

# Main simulator class with queuing and packet loss
class TunnelNetworkSimulator:
    def __init__(self, env, graph, queue_size=5):
        self.env = env
        self.graph = graph
        self.queue_size = queue_size
        self.packet_log = []
        self.device_stats = {
            node: {
                "sent": 0, "received": 0,
                "bytes_sent": 0.0, "bytes_received": 0.0,
                "dropped": 0
            }
            for node in graph.nodes
        }
        self.queues = {
            (u, v): simpy.Store(env, capacity=queue_size)
            for u, v in graph.edges
        }

    def send_packet(self, packet, path):
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            edge = (src, dst) if (src, dst) in self.queues else (dst, src)
            link = self.graph[src][dst]

            # Check link failure
            if not link.get('active', True):
                self.device_stats[packet.src]['dropped'] += 1
                return

            # Simulate queuing and packet loss
            if random.random() < link.get('drop_rate', 0.0):
                self.device_stats[packet.src]['dropped'] += 1
                return

            store = self.queues[edge]
            if len(store.items) >= store.capacity:
                self.device_stats[packet.src]['dropped'] += 1
                return

            yield store.put(packet)

            # Simulate latency and transmission time
            latency = link.get('latency', 1)
            bandwidth = link.get('bandwidth', 10)
            transmission_time = packet.size / bandwidth
            yield self.env.timeout(latency + transmission_time)

        # Packet successfully delivered
        end_time = self.env.now
        self.packet_log.append({
            "id": packet.id,
            "src": packet.src,
            "dst": packet.dst,
            "sent_at": packet.timestamp,
            "delivered_at": end_time,
            "delay": end_time - packet.timestamp,
            "size_MB": packet.size
        })
        self.device_stats[packet.src]["sent"] += 1
        self.device_stats[packet.src]["bytes_sent"] += packet.size
        self.device_stats[packet.dst]["received"] += 1
        self.device_stats[packet.dst]["bytes_received"] += packet.size

    def generate_traffic(self, rate=10):
        nodes = list(self.graph.nodes)
        while True:
            src, dst = random.sample(nodes, 2)
            packet_size = random.uniform(0.5, 4.0)
            packet = Packet(src, dst, self.env.now, size=packet_size)
            try:
                path = nx.shortest_path(self.graph, src, dst, weight='latency')
                self.env.process(self.send_packet(packet, path))
            except nx.NetworkXNoPath:
                self.device_stats[src]['dropped'] += 1
            yield self.env.timeout(random.expovariate(1.0 / rate))

# Run simulation with failure, queuing, and packet loss
def run_simulation(num_segments=8, devices_per_segment=2, sim_time=100):
    env = simpy.Environment()
    graph = create_tunnel_topology(num_segments, devices_per_segment)
    simulator = TunnelNetworkSimulator(env, graph)
    for _ in range(5):  # 5 concurrent traffic sources
        env.process(simulator.generate_traffic(rate=5))
    env.run(until=sim_time)

    packet_df = pd.DataFrame(simulator.packet_log)
    stats_df = pd.DataFrame.from_dict(simulator.device_stats, orient='index').reset_index().rename(columns={'index': 'device'})
    return graph, packet_df, stats_df

# Visualization functions
def plot_topology(graph):
    pos = nx.spring_layout(graph, seed=42)
    edge_labels = nx.get_edge_attributes(graph, 'latency')
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={k: f"{v:.1f}" for k, v in edge_labels.items()}, font_size=6)
    plt.title("Tunnel Network Topology (Latency Labels)")
    plt.show()

def plot_delay_distribution(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df['delay'], bins=15, color='orange', edgecolor='black')
    plt.xlabel("Delay (time units)")
    plt.ylabel("Number of Packets")
    plt.title("Packet Delay Distribution")
    plt.grid(True)
    plt.show()

def plot_throughput(stats_df):
    plt.figure(figsize=(10, 4))
    plt.bar(stats_df['device'], stats_df['bytes_received'], color='green', label='Bytes Received')
    plt.bar(stats_df['device'], stats_df['bytes_sent'], bottom=stats_df['bytes_received'], color='blue', label='Bytes Sent')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Bytes (MB)")
    plt.title("Throughput Per Device")
    plt.legend()
    plt.tight_layout()
    plt.show()


def unified_topology_plot(graph, stats_df):
    pos = nx.spring_layout(graph, seed=42)
    node_labels = {}
    node_colors = []
    node_sizes = []

    # Map device stats to nodes
    stats_map = stats_df.set_index('device').to_dict('index')

    for node in graph.nodes:
        stats = stats_map.get(node, {"bytes_sent": 0, "bytes_received": 0})
        sent = stats["bytes_sent"]
        received = stats["bytes_received"]
        node_labels[node] = f"{node}\nS:{sent:.1f}MB\nR:{received:.1f}MB"
        node_colors.append(received)
        node_sizes.append(500 + 50 * (sent + received))

    edge_labels = {
        (u, v): f"L:{d['latency']:.1f}, B:{d['bandwidth']:.1f}"
        for u, v, d in graph.edges(data=True)
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis
    )
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=6, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(node_colors)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Bytes Received (MB)')

    ax.set_title("Unified Network Topology View (Latency, Bandwidth, Throughput)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Animated visualization on top of simulation results
def animate_packet_flows(graph, packet_log):
    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw static network background
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='skyblue', node_size=700)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='lightgray')
    ax.set_title("Network Packet Flow Animation")
    ax.axis('off')

    # Precompute packet paths and timings
    animated_packets = []
    for _, row in packet_log.iterrows():
        try:
            path = nx.shortest_path(graph, row['src'], row['dst'], weight='latency')
            timespan = row['delivered_at'] - row['sent_at']
            animated_packets.append({
                "path": path,
                "coords": [pos[n] for n in path],
                "start": row['sent_at'],
                "end": row['delivered_at'],
                "size": row['size_MB']
            })
        except nx.NetworkXNoPath:
            continue

    max_time = packet_log['delivered_at'].max()
    frames = int(max_time * 10)  # 10 fps
    active_arrows = []

    def update(frame):
        current_time = frame / 10.0  # simulate time
        ax.clear()

        # Redraw base network
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='skyblue', node_size=700)
        nx.draw_networkx_labels(graph, pos, ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='lightgray')
        ax.set_title(f"Network Packet Flow - Time {current_time:.1f}")
        ax.axis('off')

        # Animate each packet currently in flight
        for pkt in animated_packets:
            if pkt["start"] <= current_time <= pkt["end"]:
                path_coords = pkt["coords"]
                path_progress = (current_time - pkt["start"]) / (pkt["end"] - pkt["start"])
                total_hops = len(path_coords) - 1
                hop_float = path_progress * total_hops
                hop_index = int(hop_float)
                intra_hop = hop_float - hop_index

                if hop_index < total_hops:
                    x0, y0 = path_coords[hop_index]
                    x1, y1 = path_coords[hop_index + 1]
                    x = x0 + (x1 - x0) * intra_hop
                    y = y0 + (y1 - y0) * intra_hop
                    ax.annotate("",
                        xy=(x, y), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color='red', lw=2 + pkt["size"] * 0.5, alpha=0.8))

    ani = FuncAnimation(fig, update, frames=frames, interval=100, repeat=False)
    plt.show()
    return ani




# Run everything
graph, packet_log, stats_df = run_simulation()
#plot_topology(graph)
#plot_delay_distribution(packet_log)
#plot_throughput(stats_df)
#unified_topology_plot(graph, stats_df)
ani = animate_packet_flows(graph, packet_log)


