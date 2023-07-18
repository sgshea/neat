use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use rand::Rng;
use rand::seq::IteratorRandom;
use rand::prelude::SliceRandom;

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
        if petgraph::algo::is_cyclic_directed(&self.network_graph) {
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

    // Mutate a random weight on a connection
    fn mutate_weight(&mut self) {
        let rand_edge = self.network_graph.edge_indices().choose(&mut rand::thread_rng());
        if let Some(edge) = rand_edge {
            let edge = self.network_graph.edge_weight_mut(edge).unwrap();
            edge.set_weight(rand::thread_rng().gen_range(-10.0..=10.0));
        }
    }

    // Helper function for loading inputs into proper nodes
    fn load_inputs(&mut self, inputs: Vec<f32>) {
        assert_eq!(inputs.len(), self.input_nodes);

        for (i, input) in inputs.iter().enumerate() {
            let node = self.network_graph.node_weight_mut(NodeIndex::new(i)).unwrap();
            node.update_sum(*input);
        }
    }

    pub fn output(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.load_inputs(inputs);

        Vec::new()
    }

    pub fn mutate(&mut self) {
        // TODO
    }

    pub fn output_graph(&self) {
        println!(
            "{:?}",
            Dot::with_config(&self.network_graph, &[Config::GraphContentOnly])
        );
    }
}
