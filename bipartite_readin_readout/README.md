# Bipartite Read-in and Read-out

## Introduction

### What Are They?
Bipartite Read-in and Bipartite Read-out are mechanisms used in Graph Neural Networks (GNNs) to aggregate and transform information in bipartite graphs. A bipartite graph contains two disjoint sets of nodes (e.g., \(U\) and \(V\)), where edges only exist between nodes in different sets.

- **Read-in**: Aggregates information from one set of nodes (e.g., \(U\)) to the other set (e.g., \(V\)).
- **Read-out**: Propagates aggregated information from the second set of nodes (\(V\)) back to the first set (\(U\)).

### Why Are They Used?
- Capture interactions between two sets of nodes in a bipartite graph.
- Transform and propagate features to enrich the representation of nodes based on their relationships.
- Facilitate tasks like classification, regression, or link prediction in domains such as recommendation systems, seismic networks, and social networks.

### How Do They Work?
- **Read-in**:
  1. Each node in \(V\) aggregates features from its connected neighbors in \(U\).
  2. Aggregation functions like `mean`, `sum`, or `max` are used.
- **Read-out**:
  1. Each node in \(U\) aggregates features from its connected neighbors in \(V\).
  2. The process is similar to Read-in but with reversed edge directions.

---

## Application in Seismic Phase Association

In the article, Bipartite Read-in and Read-out are used to process the interactions between seismic sources (\(U\)) and stations (\(V\)).

- **Bipartite Read-in**: Aggregates information from source-station pairs to create a spatial representation of seismic events.
- **Bipartite Read-out**: Projects information back from the spatial domain (processed by Read-in) to the source-station domain, refining predictions regarding source-arrival associations.

These mechanisms allow efficient localization and classification of seismic events by combining spatial and station-specific insights.

---

## Implementation

### Graph Structure
- **Nodes**: \(U\) (10 nodes) and \(V\) (15 nodes) with random features.
- **Edges**: 20 edges connect \(U\)-nodes to \(V\)-nodes.

### Read-in Mechanism
- \(V\)-nodes aggregate features from \(U\)-nodes using the `mean` aggregation function.

### Read-out Mechanism
- \(U\)-nodes aggregate the transformed features from \(V\)-nodes using the reverse edge connections.

### Aggregation Results
- The output features were stored for analysis after both Read-in and Read-out steps.

---

## Results

### Initial Features
- Nodes in \(U\) and \(V\) were assigned random features, representing independent properties.

### Aggregated Features After Read-in
- Features of \(V\)-nodes became influenced by their connected \(U\)-nodes. Example:
  - \(V_1\): Initial: `[0.2386, 0.7470, 0.2598, 0.9649, 0.8275]`
  - After Read-in: `[0.8821, 0.6214, 0.8496, 0.5233, 0.4903]`

### Aggregated Features After Read-out
- Features of \(U\)-nodes became influenced by the aggregated \(V\)-nodes. Example:
  - \(U_2\): Initial: `[0.0134, 0.1655, 0.8907, 0.2848, 0.6863]`
  - After Read-out: `[0.2226, 0.2706, 0.8575, 0.3972, 0.5888]`

### Key Observations
1. Feature propagation resulted in smoother values, showing the integration of neighborhood information.
2. Densely connected nodes showed stronger feature aggregation.

---

## Conclusion

Bipartite Read-in and Read-out are powerful methods for processing bipartite graphs, enabling effective feature transformation and propagation. Their implementation on synthetic data revealed their ability to enrich node representations, and their application in seismic networks demonstrated their utility in real-world scenarios.
