use std::collections::HashMap;

use petgraph::{graph::NodeIndex, stable_graph};

use super::genome::Genome;

// Creates a petgraph graph from a genome
pub fn generate_graph(genome: &Genome) -> stable_graph::StableGraph<(), ()> {
    let mut graph = petgraph::stable_graph::StableGraph::new();
    // Index node id -> graph id
    let mut ids: HashMap<usize, NodeIndex> = HashMap::new();
    for (node_id, _) in genome.nodes.iter() {
        let graph_id = graph.add_node(());
        ids.insert(*node_id, graph_id);
    }
    for (_, connection) in genome.connections.iter() {
        let source_id = ids[&connection.in_node];
        let target_id = ids[&connection.out_node];
        graph.add_edge(source_id, target_id, ());
    }
    graph
}
