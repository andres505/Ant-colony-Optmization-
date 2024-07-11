# Ant-colony-Optmization-
 Ant Colony Optimization (ACO) to find the best route to take within the Mexico City metro system. The project leverages this bio-inspired algorithm to simulate the behavior of ants finding optimal paths, providing an efficient solution for navigating the complex metro network.
# Import Libraries 
```python
import matplotlib.pyplot as plt
import networkx as nx
import random
```
# Creating the graph
To use this code, you need to create the graph representing the metro system. This involves: 
+ Labeling the nodes (metro stations).
+ Defining the edges between the nodes (connections between stations).
+ Providing the distances between the nodes (distance or travel time between connected stations).



## Create the graph
```python
G = nx.Graph()
```
## Add edges to the graph
```python
G.add_edge("El rosario", "Tacuba", weight=5.264)
G.add_edge("Tacuba", "Tacubaya", weight=5.843)
G.add_edge("Martin Carrera", "D18M", weight=1.711)
G.add_edge("D18M", "IPetro", weight=2.333)
G.add_edge( "IPetro","D18M", weight=2.333)
G.add_edge("El rosario", "IPetro", weight=5.890)
G.add_edge("Martin Carrera", "Consulado", weight=2.733)
G.add_edge("Consulado", "Morelos", weight=1.794)
G.add_edge("Morelos", "Candelaria", weight=1.062)
G.add_edge("Candelaria", "Jamaica", weight=2.726)
G.add_edge("Jamaica", "Santa Anita", weight=0.758)
G.add_edge("Tacubaya", "CM", weight=3.240)
G.add_edge("CM", "Chabacano", weight=2.059)
G.add_edge("Chabacano", "Jamaica", weight=1.031)
G.add_edge("Jamaica", "Pantitlan", weight=5.053)
G.add_edge("Tacuba", "Hidalgo", weight=4.016)
G.add_edge("Bellas", "Hidalgo", weight=0.447)
G.add_edge("Bellas", "PinoS", weight=1.734)
G.add_edge("PinoS", "Chabacano", weight=1.459)
G.add_edge("Tacubaya", "Balderas", weight=4.479)
G.add_edge("Balderas", "SaltoA", weight=0.458)
G.add_edge("SaltoA", "PinoS", weight=0.827)
G.add_edge("PinoS", "Candelaria", weight=1.443)
G.add_edge("Candelaria", "SanLazaro", weight=0.866)
G.add_edge("SanLazaro", "Pantitlan", weight=4.469)
G.add_edge("IPetro", "La raza", weight=2.042)
G.add_edge("La raza", "Consulado", weight=2.540)
G.add_edge("Consulado", "Oceania", weight=2.894)
G.add_edge("Oceania", "Pantitlan", weight=3.971)
G.add_edge("Guerrero", "Garibaldi", weight=0.757)
G.add_edge("Garibaldi", "Morelos", weight=1.583)
G.add_edge("Morelos", "SanLazaro", weight=1.296)
G.add_edge("SanLazaro", "Oceania", weight=2.624)
G.add_edge("Garibaldi", "Bellas", weight=0.634)
G.add_edge("Bellas", "SaltoA", weight=0.748)
G.add_edge("SaltoA", "Chabacano", weight=2.468)
G.add_edge("Chabacano", "Santa Anita", weight=1.476)
G.add_edge("D18M", "La raza", weight=2.072)
G.add_edge("La raza", "Guerrero", weight=2.487)
G.add_edge("Guerrero", "Hidalgo", weight=0.702)
G.add_edge("Hidalgo", "Balderas", weight=0.910)
G.add_edge("Balderas", "CM", weight=1.877)

```
# Plot the graph
```python
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < .8]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.8]

pos = nx.spring_layout(G, seed=5)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=200)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=4, edge_color="r")
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
# edge weight labels
#edge_labels = nx.get_edge_attributes(G, "weight")
#nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()

```
 
# Creating the functions for ACO
## Main function that returns the best path
```python
def ant_colony_optimization(graph, source, destination, num_ants, num_iterations, evaporation_rate, alpha, beta):
    pheromone = initialize_pheromone(graph)

    best_path = None
    best_path_length = float('inf')

    for _ in range(num_iterations):
        paths = []
        path_lengths = []

        for _ in range(num_ants):
            path = construct_path(graph, source, destination, pheromone, alpha, beta)
            path_length = calculate_path_length(graph, path)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

            paths.append(path)
            path_lengths.append(path_length)

        update_pheromone(pheromone, paths, path_lengths, evaporation_rate)
    print(best_path)
    return best_path, pheromone
```
## Initializing the pheromone
```python
def initialize_pheromone(graph):

    pheromone = {}
    for edge in graph.edges:
        pheromone[edge] = 1.0
        #Para que reivse los dos lados
        reverse_edge = (edge[1], edge[0])
        pheromone[reverse_edge] = 1.0
    return pheromone
```
## Construction of the path based on the graph
```python
def construct_path(graph, source, destination, pheromone, alpha, beta):
    path = [source]
    current_node = source

    while current_node != destination:
        next_node = select_next_node(graph, current_node, destination, pheromone, alpha, beta)
        path.append(next_node)
        current_node = next_node

    return path
```
## Randomly selecting the next node based on the pheromone
```python
def select_next_node(graph, current_node, destination, pheromone, alpha, beta):
    neighbors = list(graph.neighbors(current_node))

    if destination in neighbors:
        return destination

    probabilities = []

    for neighbor in neighbors:
        pheromone_value = pheromone[(current_node, neighbor)]
        distance = graph.edges[(current_node, neighbor)]['weight']
        attractiveness = (pheromone_value ** alpha) * ((1.0 / distance) ** beta)
        probabilities.append(attractiveness)

    total_prob = sum(probabilities)
    probabilities = [prob / total_prob for prob in probabilities]
    return random.choices(neighbors, probabilities)[0]
```
## Calcultating the path
```python
def calculate_path_length(graph, path):
    path_length = 0
    for i in range(len(path) - 1):
        source = path[i]
        destination = path[i+1]
        path_length += graph.edges[(source, destination)]['weight']
    return path_length
```
## Update oh the pheromone based on the paths
```python
def update_pheromone(pheromone, paths, path_lengths, evaporation_rate):
    for edge in pheromone:
        pheromone[edge] *= (1.0 - evaporation_rate)

    for path, length in zip(paths, path_lengths):
        for i in range(len(path) - 1):
            source = path[i]
            destination = path[i+1]

            pheromone[(source, destination)] += 1.0 / length
            pheromone[(destination, source)] += 1.0 / length

```
