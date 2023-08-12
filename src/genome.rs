use crate::genes::{ActivationFunction, ConnectionGene, NodeGene, NodeType};
use crate::innovation_record::InnovationRecord;
use rand::Rng;
use std::cmp::{max, Ordering};
use std::fmt::Display;

#[derive(Clone, Debug)]
pub struct Genome {
    pub genes: Vec<ConnectionGene>,
    pub node: Vec<NodeGene>,
    // Includes bias node
    inputs: usize,
    // Bias node index
    bias_node: usize,

    outputs: usize,
    layers: usize,

    pub fitness: f64,
}

impl Genome {
    pub fn new(inputs: usize, outputs: usize, innovation_record: &mut InnovationRecord) -> Self {
        let mut genome = Self {
            genes: vec![],
            node: vec![],
            inputs: inputs + 1,
            outputs,
            layers: 2,
            bias_node: 0,
            fitness: 0.0,
        };

        for _ in 0..inputs {
            genome.node.push(NodeGene::new(
                innovation_record.new_node_innovation(),
                NodeType::Input,
                1,
                0.0,
                0.0,
            ));
        }
        // Push bias node
        let bias_id = innovation_record.new_node_innovation();
        genome
            .node
            .push(NodeGene::new(bias_id, NodeType::Bias, 1, 0.0, 0.0));
        genome.bias_node = bias_id;
        for _ in 0..outputs {
            genome.node.push(NodeGene::new(
                innovation_record.new_node_innovation(),
                NodeType::Output,
                2,
                0.0,
                0.0,
            ));
        }

        genome.fully_connect(innovation_record);
        genome
    }

    fn new_blank(inputs: usize, outputs: usize, bias_id: usize) -> Self {
        Self {
            genes: vec![],
            node: vec![],
            inputs: inputs + 1,
            outputs,
            layers: 2,
            bias_node: bias_id,
            fitness: 0.0,
        }
    }

    pub fn crossover(&mut self, other: Genome) -> Genome {
        let mut child = Genome::new_blank(self.inputs, self.outputs, self.bias_node);
        let mut rng = rand::thread_rng();

        // Genes to be inherited
        let mut child_genes: Vec<&ConnectionGene> = vec![];

        for i in 0..self.genes.len() {
            match self.matching_gene(&other, i) {
                None => {
                    child_genes.push(&self.genes[i]);
                }
                Some(gene) => {
                    if rng.gen::<f64>() < 0.5 {
                        child_genes.push(gene);
                    } else {
                        child_genes.push(&self.genes[i]);
                    }
                }
            }
        }

        // Add all of self's nodes
        for node in &self.node {
            child.node.push(node.clone());
        }
        // Add genes
        for gene in child_genes {
            child.genes.push(gene.clone());
        }

        child
    }

    // Returns matching connection gene if exists
    fn matching_gene<'a>(&'a self, other: &'a Genome, id: usize) -> Option<&ConnectionGene> {
        let gene = other.genes.iter().find(|gene| gene.innovation == id);
        gene
    }

    pub fn mutate(&mut self, innovation_record: &mut InnovationRecord) {
        let mut rng = rand::thread_rng();
        // Mutate weights 80%
        if rng.gen::<f64>() < 0.7 {
            for gene in &mut self.genes {
                gene.mutate_weight();
            }
        }
        // Mutate add node 5%
        if rng.gen::<f64>() < 0.1 {
            self.add_node(innovation_record);
        }
        // Mutate add connection 5%
        if rng.gen::<f64>() < 0.15 {
            self.add_connection(innovation_record);
        }
    }

    pub fn add_connection(&mut self, innovation_record: &mut InnovationRecord) {
        // Just try a certain amount of times to find a connection
        let mut rng = rand::thread_rng();
        'outer: for _ in 0..20 {
            // Select two nodes
            let mut node_1 = self.node[rng.gen_range(0..self.node.len())].clone();
            let mut node_2 = self.node[rng.gen_range(0..self.node.len())].clone();

            if node_1.id == node_2.id {
                continue;
            }

            if node_1.node_layer == node_2.node_layer || node_1.node_layer > node_2.node_layer {
                continue;
            }

            // Check if connection already exists
            match self
                .genes
                .iter_mut()
                .find(|gene| gene.in_node == node_1.id && gene.out_node == node_2.id)
            {
                None => {
                    // Do nothing
                }
                Some(connection) => {
                    if !connection.enabled {
                        connection.enabled = true;
                        break 'outer;
                    } else {
                        continue 'outer;
                    }
                }
            };

            // Add connection
            let connection = ConnectionGene::new(
                node_1.id,
                node_2.id,
                rng.gen_range(-5.0..5.0),
                innovation_record.new_innovation(node_1.id, node_2.id),
            );
            self.genes.push(connection);
            break 'outer;
        }
    }

    pub fn add_node(&mut self, innovation_record: &mut InnovationRecord) {
        let mut rng = rand::thread_rng();
        let genes_len = self.genes.len();
        let connection = &mut self.genes[rng.gen_range(0..genes_len)];
        connection.enabled = false;
        let node_id = innovation_record.new_node_innovation();
        // from layer
        let connection_ids: (usize, usize) = (connection.in_node, connection.out_node);
        let from_layer = get_node(connection_ids.0, &mut self.node.clone())
            .unwrap()
            .node_layer;
        self.node.push(NodeGene::new(
            node_id,
            NodeType::Hidden,
            from_layer + 1,
            0.0,
            0.0,
        ));
        self.genes.push(ConnectionGene::new(
            connection_ids.0,
            node_id,
            rng.gen_range(-5.0..5.0),
            innovation_record.new_innovation(connection_ids.0, node_id),
        ));
        self.genes.push(ConnectionGene::new(
            node_id,
            connection_ids.1,
            rng.gen_range(-5.0..5.0),
            innovation_record.new_innovation(node_id, connection_ids.1),
        ));
        // Recalculate layers
        let nodes = self.node.clone();
        let genes = self.genes.clone();
        for node in &mut self.node {
            if node.node_layer == 1 {
                continue;
            }
            node.node_layer = find_layer(&nodes, &genes, Some(node));
        }
        self.layers = self.node.iter().map(|node| node.node_layer).max().unwrap();
    }

    pub fn fully_connect(&mut self, innovation_record: &mut InnovationRecord) {
        // If there are hidden nodes
        if self.node.len() > self.inputs + self.outputs {
            for i in 0..self.inputs {
                for j in self.inputs + self.outputs..=self.node.len() {
                    self.genes.push(ConnectionGene::new(
                        self.node[i].id,
                        self.node[j].id,
                        rand::thread_rng().gen_range(-5.0..5.0),
                        innovation_record.new_innovation(i, j),
                    ));
                }
            }
            for i in self.inputs + self.outputs..=self.node.len() {
                for j in 0..self.outputs {
                    self.genes.push(ConnectionGene::new(
                        self.node[i].id,
                        self.node[self.inputs + j].id,
                        rand::thread_rng().gen_range(-5.0..5.0),
                        innovation_record.new_innovation(i, self.inputs + j),
                    ));
                }
            }
        } else {
            for i in 0..self.inputs {
                for j in 0..self.outputs {
                    self.genes.push(ConnectionGene::new(
                        self.node[i].id,
                        self.node[self.inputs + j].id,
                        rand::thread_rng().gen_range(-5.0..5.0),
                        innovation_record.new_innovation(i, self.inputs + j),
                    ));
                }
            }
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // Reset
        for node in &mut self.node {
            node.sum_inputs = 0.0;
            node.sum_outputs = 0.0;
        }
        // Set input nodes
        for i in 0..inputs.len() {
            self.node[i].sum_inputs = inputs[i];
            self.node[i].sum_outputs = inputs[i];
        }
        self.node[self.bias_node].sum_inputs = 1.0;
        self.node[self.bias_node].sum_outputs = 1.0;

        let genes = self.genes.clone();
        // Collect node ids
        let mut node_ids: Vec<usize> = vec![];
        for node in &mut self.node {
            node_ids.push(node.id);
        }

        // Loop through layers starting at 2
        for i in 2..=self.layers {
            for node_id in &node_ids {
                let mut node = get_node(*node_id, &self.node).unwrap().clone();
                if node.node_layer == i {
                    // Find all incoming connections
                    genes.iter().for_each(|gene| {
                        if gene.out_node == node.id && gene.enabled {
                            let in_node = get_node(gene.in_node, &mut self.node).unwrap();
                            node.sum_inputs += in_node.sum_outputs * gene.weight;
                        }
                    });
                    // Apply activation function
                    let node_index = self
                        .node
                        .iter()
                        .position(|node| node.id == node_id.clone())
                        .unwrap();
                    self.node[node_index].sum_inputs = node.sum_inputs;
                    self.node[node_index].sum_outputs =
                        1.0 / (1.0 + (-4.9 * node.sum_inputs).exp());
                }
            }
        }

        // Get output nodes
        let mut outputs = vec![];
        for node in &mut self.node {
            if node.node_type == NodeType::Output {
                outputs.push(node.sum_outputs);
            }
        }
        outputs
    }

    pub fn compatability_distance(&self, other: &Self) -> f64 {
        // CD = E + D + abs(W)
        // Where E is excess genes (amount with a higher inno number than exists in other)
        // D is disjoint genes (other non matching)
        let max_len = max(self.genes.len(), other.genes.len());
        // Assume called genome is fitter
        let highest_inno_1 = other.genes.last().unwrap().innovation;
        let highest_inno_2 = self.genes.last().unwrap().innovation;
        let excess_amt: usize = if highest_inno_1 > highest_inno_2 {
            self.genes
                .iter()
                .filter(|gene| gene.innovation > highest_inno_1)
                .count()
        } else {
            other
                .genes
                .iter()
                .filter(|gene| gene.innovation > highest_inno_2)
                .count()
        };

        let disjoint_1 = self
            .genes
            .iter()
            .filter(|gene| {
                other
                    .genes
                    .iter()
                    .find(|other_gene| other_gene.innovation == gene.innovation)
                    .is_none()
            })
            .count();
        let disjoint_2 = other
            .genes
            .iter()
            .filter(|gene| {
                self.genes
                    .iter()
                    .find(|other_gene| other_gene.innovation == gene.innovation)
                    .is_none()
            })
            .count();

        let same_amt = self
            .genes
            .iter()
            .filter(|gene| {
                other
                    .genes
                    .iter()
                    .find(|other_gene| other_gene.innovation == gene.innovation)
                    .is_some()
            })
            .count();

        // Average weight difference is abs(W)
        let average_weight_diff = self.genes.iter().fold(0.0, |acc, gene| {
            let other_gene = other
                .genes
                .iter()
                .find(|other_gene| other_gene.innovation == gene.innovation);
            match other_gene {
                None => acc,
                Some(other_gene) => acc + (gene.weight - other_gene.weight).abs(),
            }
        }) / same_amt as f64;

        ((excess_amt / max_len) as f64
            + ((disjoint_1 + disjoint_2) / max_len) as f64
            + 0.4 * average_weight_diff)
    }
}

fn get_node(id: usize, nodes: &Vec<NodeGene>) -> Option<&NodeGene> {
    let node = nodes.iter().find(|node| node.id == id);
    match node {
        None => None,
        Some(node) => Some(node),
    }
}

fn find_layer(
    nodes: &Vec<NodeGene>,
    genes: &Vec<ConnectionGene>,
    node: Option<&NodeGene>,
) -> usize {
    match node {
        None => 0,
        Some(node) => {
            // Get all connections to node
            let connections: Vec<&ConnectionGene> = genes
                .iter()
                .filter(|gene| gene.out_node == node.id)
                .collect();
            if connections.len() == 0 {
                return 1;
            } else {
                // Find longest path
                let mut max_layer = 0;
                for connection in connections {
                    let node_layer =
                        find_layer(&nodes, genes, get_node(connection.in_node, &nodes));
                    if node_layer > max_layer {
                        max_layer = node_layer;
                    }
                }
                max_layer + 1
            }
        }
    }
}

impl Display for Genome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();
        output.push_str(&format!("Fitness: {} ", self.fitness));
        output.push_str(&format!("Layers: {} ", self.layers));
        output.push_str(&format!("Nodes:\n"));
        for node in &self.node {
            output.push_str(&format!("{:?}\n", node));
        }
        output.push_str(&format!("Genes:\n"));
        for gene in &self.genes {
            output.push_str(&format!("{:?}\n", gene));
        }
        write!(f, "{}", output)
    }
}

impl Eq for Genome {}
impl PartialEq<Self> for Genome {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd<Self> for Genome {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}

impl Ord for Genome {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fitness.partial_cmp(&other.fitness).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_genome() {
        let mut innovation_record = InnovationRecord::new();
        let mut genome = Genome::new(2, 1, &mut innovation_record);
        assert_eq!(genome.inputs, 3);
        assert_eq!(genome.outputs, 1);
        assert_eq!(genome.layers, 2);
        assert_eq!(genome.node.len(), 4);
        assert_eq!(genome.genes.len(), 3);

        // Add a bunch of mutation
        for _ in 0..16 {
            genome.mutate(&mut innovation_record);
        }
        dbg!(genome.genes);
        dbg!(genome.node);
    }

    #[test]
    fn proper_output() {
        // Test case to make sure feed-forward has proper output
        let mut innovation_record = InnovationRecord::new();
        let mut genome = Genome::new(2, 1, &mut innovation_record);

        // Manually set all weights
        genome.genes[0].weight = 0.5;
        genome.genes[1].weight = 0.5;
        genome.genes[2].weight = 0.5;

        let output = genome.feed_forward(vec![0.0, 0.0]);
        assert_eq!(output[0], 0.6224593312018546);
        let output = genome.feed_forward(vec![1.0, 0.0]);
        assert_eq!(output[0], 0.7310585786300049);
        let output = genome.feed_forward(vec![0.0, 1.0]);
        assert_eq!(output[0], 0.7310585786300049);
        let output = genome.feed_forward(vec![1.0, 1.0]);
        assert_eq!(output[0], 0.8175744761936437);
        dbg!(genome);
    }

    #[test]
    fn compare_check() {
        // Simple comparison of genomes to make sure that sorting by fitness will work
        let mut innovation_record = InnovationRecord::new();
        let mut genome = Genome::new(2, 1, &mut innovation_record);
        genome.fitness = 5.0;
        let mut genome_2 = Genome::new(2, 1, &mut innovation_record);
        genome_2.fitness = 10.0;

        assert!(genome < genome_2);

        let mut vec = vec![genome_2, genome];
        assert_eq!(vec[1].fitness, 5.0);
        vec.sort();
        assert_eq!(vec[1].fitness, 10.0);
    }
}
