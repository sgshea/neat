use petgraph::{algo, Direction};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::IntoNeighborsDirected;
use rand::Rng;
use rand::seq::IteratorRandom;
use rand::prelude::SliceRandom;
use crate::activation::Activation;

use crate::connection::Connection;
use crate::innovation_record::InnovationRecord;
use crate::node::{Node, NodeType};

#[derive(Debug, Clone)]
pub struct Genome {
    // Directed graph
    network_graph: DiGraph<Node, Connection>,
    input_nodes: usize,
    output_nodes: usize,
    hidden_nodes: usize,
}

impl Genome {

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
            network_graph.add_node(Node::new(i + input_nodes + output_nodes + 1, NodeType::Hidden));
        }

        Self {
            network_graph,
            input_nodes,
            output_nodes,
            hidden_nodes,
        }
    }

    // Create a full graph
    pub fn new(input_nodes: usize, output_nodes: usize, hidden_nodes: usize, innovation_record: &mut InnovationRecord) -> Self {
        let mut genome = Self::new_disconnected(input_nodes, output_nodes, hidden_nodes);

        for i in 0..input_nodes {
            for j in 0..hidden_nodes {
                let innovation_number = innovation_record.new_connection();
                genome.network_graph.add_edge(
                    NodeIndex::new(i),
                    NodeIndex::new(input_nodes + output_nodes + j + 1),
                    Connection::new(innovation_number, 1.0),
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
                        Connection::new(innovation_number, 1.0),
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
                        Connection::new(innovation_number, 1.0),
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
        let possible_conn: Vec<(NodeIndex, NodeIndex)> = network_graph.node_indices()
            .flat_map(|node| {
            let neighbors: Vec<NodeIndex> = network_graph.neighbors(node).collect();
            let node_type = network_graph.node_weight(node).unwrap().get_type();
            let unconnected_nodes = network_graph.node_indices()
                .filter(move |&n| {
                    // Conditions to restrict connections
                    let unconnected_node = network_graph.node_weight(n).unwrap().get_type();
                    let ans = match node_type {
                        NodeType::Input => unconnected_node != NodeType::Input && unconnected_node != NodeType::Bias,
                        NodeType::Output => false, // output cannot connect to anything
                        NodeType::Hidden => unconnected_node != NodeType::Hidden && unconnected_node != NodeType::Bias && unconnected_node != NodeType::Input,
                        NodeType::Bias => unconnected_node != NodeType::Bias && unconnected_node != NodeType::Input,
                    };
                    ans && n != node && !neighbors.contains(&n)
                });
            unconnected_nodes.map(move |unconnected_node| (node, unconnected_node))
        })
        .collect();

        println!("{:?}", possible_conn);

        // We will try to add connection and if it creates a cycle, remove it
        let (from, to) = possible_conn.choose(&mut rand::thread_rng()).unwrap();
        let new_connection = self.network_graph.add_edge(*from, *to, Connection::new(innovation_record.new_connection(), 1.0));
        if algo::is_cyclic_directed(&self.network_graph) {
            self.network_graph.remove_edge(new_connection);
            innovation_record.remove_last_connection();
        }
    }

    // Mutation where node is added
    fn add_node(&mut self, innovation_record: &mut InnovationRecord) {
        // Choose random edge
        let rand_edge = self.network_graph.edge_indices().choose(&mut rand::thread_rng());

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
            self.network_graph.add_edge(
                from,
                new_node,
                Connection::new(innovation_id, 1.0),
            );

            let innovation_id = innovation_record.new_connection();
            // Output edge gets weight of old edge
            self.network_graph.add_edge(
                new_node,
                to,
                Connection::new(innovation_id, self.network_graph.edge_weight(edge).unwrap().get_weight()),
            );

            self.network_graph.remove_edge(edge);
        }
    }

    // Helper function for loading inputs into proper nodes
    fn load_inputs(&mut self, inputs: &[f32]) {
        assert_eq!(inputs.len(), self.input_nodes);

        for (i, input) in inputs.iter().enumerate() {
            let node = self.network_graph.node_weight_mut(NodeIndex::new(i)).unwrap();
            node.update_sum(*input);
            node.activate(Activation::Linear);
        }
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
        let output_nodes: Vec<f32> = graph.node_indices()
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
                if rng.gen::<f32>() < 0.9 {
                    // Add random value to weight
                    conn.set_weight(conn.get_weight() + rng.gen_range(-0.2..=0.2));
                }
                else {
                    // Set to random value
                    conn.set_weight(rng.gen_range(-2.0..=2.0));
                }
            }
        }
        if rng.gen::<f32>() < 0.003 {
            // split connection (add_node)
            self.add_node(innovation_record);
        }
        if rng.gen::<f32>() <0.01 {
            // swap connection (enable/disable)
        }
        if rng.gen::<f32>() < 0.005 {
            // add connection
            self.add_connection(innovation_record);
        }
    }

    pub fn output_graph(&self) {
        println!(
            "{:?}",
            Dot::with_config(
                &self.network_graph,
                &[Config::GraphContentOnly]
            )
        );
    }
}
