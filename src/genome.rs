use crate::activation::Activation;
use petgraph::data::DataMap;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::{algo, Direction};
use rand::prelude::SliceRandom;
use rand::seq::IteratorRandom;
use rand::Rng;
use std::collections::HashMap;

use crate::connection::Connection;
use crate::innovation_record::InnovationRecord;
use crate::node::{Node, NodeType};

#[derive(Debug, Clone)]
pub struct Genome {
    // Directed graph
    pub network_graph: DiGraph<Node, Connection>,
    input_nodes: usize,
    output_nodes: usize,
    hidden_nodes: usize,
}

impl Genome {
    fn new_from_graph(
        network_graph: DiGraph<Node, Connection>,
        input_nodes: usize,
        output_nodes: usize,
        hidden_nodes: usize,
    ) -> Self {
        Self {
            network_graph,
            input_nodes,
            output_nodes,
            hidden_nodes,
        }
    }

    // Create a disconnected graph
    pub fn new_disconnected(input_nodes: usize, output_nodes: usize, hidden_nodes: usize) -> Self {
        let mut network_graph = DiGraph::<Node, Connection>::new();

        for i in 0..input_nodes {
            network_graph.add_node(Node::new(i, NodeType::Input));
        }

        network_graph.add_node(Node::new(input_nodes, NodeType::Bias));

        for i in 0..output_nodes {
            network_graph.add_node(Node::new(i + input_nodes + 1, NodeType::Output));
        }

        for i in 0..hidden_nodes {
            network_graph.add_node(Node::new(
                i + input_nodes + output_nodes + 1,
                NodeType::Hidden,
            ));
        }

        Self {
            network_graph,
            input_nodes,
            output_nodes,
            hidden_nodes,
        }
    }

    // Create a full graph
    pub fn new(
        input_nodes: usize,
        output_nodes: usize,
        hidden_nodes: usize,
        innovation_record: &mut InnovationRecord,
    ) -> Self {
        let mut genome = Self::new_disconnected(input_nodes, output_nodes, hidden_nodes);

        let rng = &mut rand::thread_rng();

        for i in 0..input_nodes {
            for j in 0..hidden_nodes {
                let innovation_number = innovation_record.new_connection();
                genome.network_graph.add_edge(
                    NodeIndex::new(i),
                    NodeIndex::new(input_nodes + output_nodes + j + 1),
                    Connection::new(innovation_number, rng.gen_range(-2.0..2.0)),
                );
            }
        }

        if hidden_nodes > 0 {
            for i in 0..hidden_nodes {
                for j in 0..output_nodes {
                    let innovation_number = innovation_record.new_connection();
                    genome.network_graph.add_edge(
                        NodeIndex::new(input_nodes + output_nodes + i + 1),
                        NodeIndex::new(input_nodes + j + 1),
                        Connection::new(innovation_number, rng.gen_range(-2.0..2.0)),
                    );
                }
            }
        } else {
            for i in 0..input_nodes {
                for j in 0..output_nodes {
                    let innovation_number = innovation_record.new_connection();
                    genome.network_graph.add_edge(
                        NodeIndex::new(i),
                        NodeIndex::new(input_nodes + j + 1),
                        Connection::new(innovation_number, rng.gen_range(-2.0..2.0)),
                    );
                }
            }
        }

        genome
    }

    // Mutation where connection is added
    fn add_connection(&mut self, innovation_record: &mut InnovationRecord) {
        // Get possible connections
        let network_graph = &self.network_graph.clone();

        // This should probably be changed TODO
        // But it works
        let possible_conn: Vec<(NodeIndex, NodeIndex)> = network_graph
            .node_indices()
            .flat_map(|node| {
                let neighbors: Vec<NodeIndex> = network_graph.neighbors(node).collect();
                let node_type = network_graph.node_weight(node).unwrap().get_type();
                let unconnected_nodes = network_graph.node_indices().filter(move |&n| {
                    // Conditions to restrict connections
                    let unconnected_node = network_graph.node_weight(n).unwrap().get_type();
                    let ans = match node_type {
                        NodeType::Input => {
                            unconnected_node != NodeType::Input
                                && unconnected_node != NodeType::Bias
                        }
                        NodeType::Output => false, // output cannot connect to anything
                        NodeType::Hidden => {
                            unconnected_node != NodeType::Hidden
                                && unconnected_node != NodeType::Bias
                                && unconnected_node != NodeType::Input
                        }
                        NodeType::Bias => {
                            unconnected_node != NodeType::Bias
                                && unconnected_node != NodeType::Input
                        }
                    };
                    ans && n != node && !neighbors.contains(&n)
                });
                unconnected_nodes.map(move |unconnected_node| (node, unconnected_node))
            })
            .collect();

        // We will try to add connection and if it creates a cycle, remove it
        match possible_conn.choose(&mut rand::thread_rng()) {
            None => {} // Do nothing
            Some((from, to)) => {
                let new_connection = self.network_graph.add_edge(
                    *from,
                    *to,
                    Connection::new(innovation_record.new_connection(), 1.0),
                );
                if algo::is_cyclic_directed(&self.network_graph) {
                    self.network_graph.remove_edge(new_connection);
                    innovation_record.remove_last_connection();
                }
            }
        };
    }

    // Mutation where node is added
    fn add_node(&mut self, innovation_record: &mut InnovationRecord) {
        // Choose random edge
        let rand_edge = self
            .network_graph
            .edge_indices()
            .choose(&mut rand::thread_rng());

        // Split connection into two, adding node in between
        if let Some(edge) = rand_edge {
            // Get NodeIndex (from, to) of edge
            let (from, to) = self.network_graph.edge_endpoints(edge).unwrap();

            let innovation_id = innovation_record.new_node();

            // Create new node
            let new_node = self
                .network_graph
                .add_node(Node::new(innovation_id, NodeType::Hidden));

            let innovation_id = innovation_record.new_connection();
            // Create new edges
            // Input edge gets weight of 1.0
            self.network_graph
                .add_edge(from, new_node, Connection::new(innovation_id, 1.0));

            let innovation_id = innovation_record.new_connection();
            // Output edge gets weight of old edge
            self.network_graph.add_edge(
                new_node,
                to,
                Connection::new(
                    innovation_id,
                    self.network_graph.edge_weight(edge).unwrap().get_weight(),
                ),
            );

            self.network_graph.remove_edge(edge);
        }
    }

    // Helper function
    // Collects genes that are matching, disjoint between self (fitter) and other genome
    // Also returns all connections for each for further use
    fn difference(
        &self,
        other: &Genome,
    ) -> (
        HashMap<usize, EdgeIndex>,
        HashMap<usize, EdgeIndex>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
    ) {
        // First collect all connections from each
        let self_conn: HashMap<usize, EdgeIndex> = self
            .network_graph
            .edge_indices()
            .map(|edge| {
                (
                    self.network_graph
                        .edge_weight(edge)
                        .unwrap()
                        .get_innovation_id(),
                    edge,
                )
            })
            .collect();

        let other_conn: HashMap<usize, EdgeIndex> = other
            .network_graph
            .edge_indices()
            .map(|edge| {
                (
                    other
                        .network_graph
                        .edge_weight(edge)
                        .unwrap()
                        .get_innovation_id(),
                    edge,
                )
            })
            .collect();

        // Get matching genes
        let matching_genes: Vec<usize> = self_conn
            .keys()
            .filter(|key| other_conn.contains_key(key))
            .map(|key| *key)
            .collect();

        // Get disjoint genes
        let disjoint_genes: Vec<usize> = self_conn
            .keys()
            .filter(|key| !other_conn.contains_key(key))
            .map(|key| *key)
            .collect();

        // Get other's disjoint genes
        let other_disjoint: Vec<usize> = other_conn
            .keys()
            .filter(|key| !self_conn.contains_key(key))
            .map(|key| *key)
            .collect();

        (
            self_conn,
            other_conn,
            matching_genes,
            disjoint_genes,
            other_disjoint,
        )
    }

    // crossover function
    pub fn crossover(&self, other: &Genome) -> Genome {
        let mut rng = rand::thread_rng();

        let (self_conn, other_conn, matching_genes, disjoint_genes, _) = self.difference(other);

        let mut new_genes: Vec<(Connection, &Node, &Node)> = Vec::new();

        // Random chance (50%)
        for i in matching_genes {
            // Choose a random connection gene to use and place in new_genes
            let (from, to) = self.network_graph.edge_endpoints(self_conn[&i]).unwrap();
            if rng.gen::<f32>() <= 0.50 {
                new_genes.push((
                    self.network_graph
                        .edge_weight(self_conn[&i])
                        .unwrap()
                        .clone(),
                    self.network_graph.node_weight(from).unwrap(),
                    self.network_graph.node_weight(to).unwrap(),
                ));
            } else {
                let (from, to) = self.network_graph.edge_endpoints(self_conn[&i]).unwrap();
                new_genes.push((
                    other
                        .network_graph
                        .edge_weight(other_conn[&i])
                        .unwrap()
                        .clone(),
                    self.network_graph.node_weight(from).unwrap(),
                    self.network_graph.node_weight(to).unwrap(),
                ));
            }
        }

        // Now handle disjoint/excess genes
        // We assume the genome being called is the more fit one and take it's excess genes
        for i in disjoint_genes {
            let (from, to) = self.network_graph.edge_endpoints(self_conn[&i]).unwrap();
            new_genes.push((
                self.network_graph
                    .edge_weight(self_conn[&i])
                    .unwrap()
                    .clone(),
                self.network_graph.node_weight(from).unwrap(),
                self.network_graph.node_weight(to).unwrap(),
            ));
        }

        // Create new graph
        let mut new_graph = DiGraph::<Node, Connection>::new();

        // Keep track of adding new nodes
        let mut added_nodes: HashMap<usize, NodeIndex> = HashMap::new();

        // Add input/output/bias nodes
        for node in self.network_graph.raw_nodes() {
            if node.weight.get_type() == NodeType::Input
                || node.weight.get_type() == NodeType::Output
                || node.weight.get_type() == NodeType::Bias
            {
                let new_node =
                    new_graph.add_node(Node::new(node.weight.get_id(), node.weight.get_type()));
                added_nodes.insert(node.weight.get_id(), new_node);
            }
        }

        // Add new genes to graph
        for (connection, node1, node2) in new_genes {
            let from = if added_nodes.contains_key(&node1.get_id()) {
                added_nodes[&node1.get_id()]
            } else {
                let new_node = new_graph.add_node(Node::new(node1.get_id(), node1.get_type()));
                added_nodes.insert(node1.get_id(), new_node);
                new_node
            };

            let to = if added_nodes.contains_key(&node2.get_id()) {
                added_nodes[&node1.get_id()]
            } else {
                let new_node = new_graph.add_node(Node::new(node2.get_id(), node2.get_type()));
                added_nodes.insert(node2.get_id(), new_node);
                new_node
            };

            new_graph.add_edge(from, to, connection);
        }

        Genome::new_from_graph(
            new_graph,
            self.input_nodes,
            self.output_nodes,
            self.hidden_nodes,
        )
    }

    pub fn compatability_distance(&self, other: &Genome) -> f64 {
        let (self_conn, other_conn, matching_genes, disjoint_genes, other_disjoint) =
            self.difference(other);

        let mut weight_diff: f64 = 0.0;

        for i in &matching_genes {
            weight_diff += (self
                .network_graph
                .edge_weight(self_conn[&i])
                .unwrap()
                .get_weight() as f64
                - other
                    .network_graph
                    .edge_weight(other_conn[&i])
                    .unwrap()
                    .get_weight() as f64)
                .abs();
        }

        let n = f64::max(self_conn.len() as f64, other_conn.len() as f64);

        // compatibility distance formula is (c1 * E) / N + (c2 * D) / N + c3 * W
        // page 13 of NEAT paper
        // E = number of disjoint genes from self
        // D = number of disjoint genes from other
        // N = number of genes in larger genome
        // W = average weight difference of matching genes
        // c1, c2, c3 are configurable coefficients
        // TODO: Configurable coefficients

        (disjoint_genes.len() as f64 / n)
            + (other_disjoint.len() as f64 / n)
            + (weight_diff / matching_genes.len() as f64)
    }

    // Helper function for loading inputs into proper nodes
    fn load_inputs(&mut self, inputs: &[f32]) {
        assert_eq!(inputs.len(), self.input_nodes);

        for (i, input) in inputs.iter().enumerate() {
            let node = self
                .network_graph
                .node_weight_mut(NodeIndex::new(i))
                .unwrap();
            node.update_sum(*input);
            node.activate(Activation::Linear);
        }

        // Load bias node
        let bias_node = self
            .network_graph
            .node_weight_mut(NodeIndex::new(self.input_nodes))
            .unwrap();

        bias_node.update_sum(1.0);
        bias_node.activate(Activation::Linear);
    }

    // feed-forward
    pub fn output(&mut self, inputs: &[f32], activation: Activation) -> Vec<f32> {
        self.load_inputs(inputs);

        let topo = algo::toposort(&self.network_graph, None).unwrap().clone();
        let graph = &mut self.network_graph;
        for node in topo {
            // Skip input nodes
            if graph.neighbors_directed(node, Direction::Incoming).count() == 0 {
                continue;
            }

            let weighted_sum = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|n| {
                    let edge = graph.find_edge(n, node).unwrap();
                    let edge_weight = graph.edge_weight(edge).unwrap();
                    edge_weight.get_weight() * graph.node_weight(n).unwrap().get_output()
                })
                .sum();

            // apply weighted sum to node
            let node = graph.node_weight_mut(node).unwrap();
            node.update_sum(weighted_sum);
            node.activate(activation);
        }

        // Get output nodes
        let output_nodes: Vec<f32> = graph
            .node_indices()
            .filter(|&n| graph.node_weight(n).unwrap().get_type() == NodeType::Output)
            .map(|n| graph.node_weight(n).unwrap().get_sum())
            .collect();

        output_nodes
    }

    // Handle random mutations
    // Should refactor weights out to a config later (maybe pass in as parameter)
    pub fn mutate(&mut self, innovation_record: &mut InnovationRecord) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < 0.8 {
            // mutate connections
            for conn in self.network_graph.edge_weights_mut() {
                if rng.gen::<f32>() < 0.7 {
                    // Add random value to weight
                    conn.set_weight(conn.get_weight() + rng.gen_range(-0.3..=0.3));
                } else {
                    // Set to random value
                    conn.set_weight(rng.gen_range(-2.0..=2.0));
                }
            }
        }
        if rng.gen::<f32>() < 0.003 {
            // split connection (add_node)
            self.add_node(innovation_record);
        }
        if rng.gen::<f32>() < 0.01 {
            // swap connection (enable/disable)
            let mut rng = rand::thread_rng();
            let choice = self
                .network_graph
                .edge_weights_mut()
                .choose(&mut rng)
                .unwrap();
            choice.swap_enabled();
        }
        if rng.gen::<f32>() < 0.005 {
            // add connection
            self.add_connection(innovation_record);
        }
    }

    pub fn output_graph(&self) {
        println!(
            "{:?}",
            Dot::with_config(&self.network_graph, &[Config::GraphContentOnly])
        );
    }
}
