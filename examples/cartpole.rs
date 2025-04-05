use eframe::{egui, App, Frame};
use egui_graphs::{
    DefaultEdgeShape, DefaultNodeShape, LayoutHierarchical, LayoutStateHierarchical,
};
use neat::{
    context::{ActivationFunction, Environment, NeatConfig},
    genome::{genome::Genome, visualization::generate_graph},
    nn::{
        feedforward::FeedforwardNetwork,
        nn::{NetworkType, NeuralNetwork},
    },
    population::Population,
};
use std::time::{Duration, Instant};

/// Simulates cartpole dynamics using Euler integration.
/// network takes 4 inputs: cart x, cart velocity, pole angle, and pole angular velocity.
/// outputs a force direction to keep the pole balanced.
/// final fitness is the number of simulation steps the pole remains balanced.
fn cartpole_test(genome: &Genome) -> f32 {
    let mut nn = FeedforwardNetwork::new(genome).unwrap();

    let dt = 0.02; // seconds
    let gravity = 9.8;
    let mass_cart = 1.0;
    let mass_pole = 0.1;
    let pole_length = 0.5;
    let force_mag = 10.0;
    let max_steps = 500;

    let (mut x, mut x_dot, mut theta, mut theta_dot) = (0.0, 0.0, 0.05, 0.0);

    let mut steps = 0;
    for _ in 0..max_steps {
        let inputs = vec![x, x_dot, theta, theta_dot];
        let output = nn.activate(&inputs).unwrap();
        let force = if output[0] > 0.5 {
            force_mag
        } else {
            -force_mag
        };

        let costheta = theta.cos();
        let sintheta = theta.sin();
        let temp = (force + mass_pole * pole_length * theta_dot.powi(2) * sintheta)
            / (mass_cart + mass_pole);
        let theta_acc = (gravity * sintheta - costheta * temp)
            / (pole_length * (4.0 / 3.0 - mass_pole * costheta.powi(2) / (mass_cart + mass_pole)));
        let x_acc = temp - mass_pole * pole_length * theta_acc * costheta / (mass_cart + mass_pole);

        x += dt * x_dot;
        x_dot += dt * x_acc;
        theta += dt * theta_dot;
        theta_dot += dt * theta_acc;
        steps += 1;

        if x.abs() > 2.4 || theta.abs() > 0.20944 {
            break;
        }
    }
    steps as f32
}

/// This EGUI application displays the real-time cartpole simulation (left pane)
/// and a network visualization of the best genome (right pane).
struct SimulationApp<'n> {
    // Genome and network derived from the best genome.
    genome: Genome,
    network: FeedforwardNetwork<'n>,
    graph: egui_graphs::Graph,
    // Cartpole simulation state.
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    // Physics parameters.
    dt: f32,
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    pole_length: f32, // half-length of pole
    force_mag: f32,
    last_update: Instant,
}

impl<'n> SimulationApp<'n> {
    fn new(genome: Genome, network: FeedforwardNetwork<'n>) -> Self {
        let graph = egui_graphs::Graph::from(&generate_graph(&genome));
        SimulationApp {
            genome,
            graph,
            network,
            x: 0.0,
            x_dot: 0.0,
            theta: 0.05,
            theta_dot: 0.0,
            dt: 0.02,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            pole_length: 0.5,
            force_mag: 10.0,
            last_update: Instant::now(),
        }
    }

    /// Update the cartpole simulation using Euler integration.
    fn update_simulation(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_update) < Duration::from_secs_f32(self.dt) {
            return;
        }
        self.last_update = now;

        let inputs = vec![self.x, self.x_dot, self.theta, self.theta_dot];
        let output = self.network.activate(&inputs).unwrap();
        let force = if output[0] > 0.5 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let costheta = self.theta.cos();
        let sintheta = self.theta.sin();
        let temp = (force + self.mass_pole * self.pole_length * self.theta_dot.powi(2) * sintheta)
            / (self.mass_cart + self.mass_pole);
        let theta_acc = (self.gravity * sintheta - costheta * temp)
            / (self.pole_length
                * (4.0 / 3.0
                    - self.mass_pole * costheta.powi(2) / (self.mass_cart + self.mass_pole)));
        let x_acc = temp
            - self.mass_pole * self.pole_length * theta_acc * costheta
                / (self.mass_cart + self.mass_pole);

        self.x += self.dt * self.x_dot;
        self.x_dot += self.dt * x_acc;
        self.theta += self.dt * self.theta_dot;
        self.theta_dot += self.dt * theta_acc;

        // Reset simulation if the pole falls or cart leaves the bounds.
        if self.x.abs() > 2.4 || self.theta.abs() > 0.20944 {
            self.x = 0.0;
            self.x_dot = 0.0;
            self.theta = 0.05;
            self.theta_dot = 0.0;
        }
    }

    /// Render the cartpole simulation.
    /// This draws a cart as a rectangle and a pole as a line.
    fn draw_simulation(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        // Set parameters for the cart.
        let cart_y = rect.center().y + 50.0;
        let scale = rect.width() / 6.0;
        let cart_w = 50.0;
        let cart_h = 30.0;
        let sim_to_screen_x = |x: f32| rect.center().x + x * scale;
        let cart_x = sim_to_screen_x(self.x) - cart_w / 2.0;
        let cart_rect = egui::Rect::from_min_size(
            egui::pos2(cart_x, cart_y - cart_h / 2.0),
            egui::vec2(cart_w, cart_h),
        );
        painter.rect_filled(cart_rect, 4.0, egui::Color32::DARK_GRAY);

        // Draw the pole.
        let cart_center_top = egui::pos2(sim_to_screen_x(self.x), cart_y - cart_h / 2.0);
        let pole_length_px = self.pole_length * scale * 2.0;
        let pole_end = egui::pos2(
            cart_center_top.x + pole_length_px * self.theta.sin(),
            cart_center_top.y - pole_length_px * self.theta.cos(),
        );
        painter.line_segment(
            [cart_center_top, pole_end],
            egui::Stroke {
                width: 4.0,
                color: egui::Color32::from_rgb(200, 50, 50),
            },
        );
        painter.circle_filled(pole_end, 6.0, egui::Color32::from_rgb(200, 50, 50));
    }
}

impl App for SimulationApp<'_> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        // Update simulation state
        self.update_simulation();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cartpole Simulation & Genome");
            ui.separator();

            // Create a horizontal layout dividing simulation and genome view
            ui.horizontal(|ui| {
                // Left pane: simulation
                ui.vertical(|ui| {
                    ui.label("Simulation");
                    let sim_rect = ui.allocate_space(egui::Vec2::new(400.0, 400.0));
                    self.draw_simulation(ui, sim_rect.1);
                });

                ui.separator();

                // Right pane: genome network visualization
                ui.vertical(|ui| {
                    ui.label("Genome Network");
                    ui.add(&mut egui_graphs::GraphView::<
                        _,
                        _,
                        _,
                        _,
                        DefaultNodeShape,
                        DefaultEdgeShape,
                        LayoutStateHierarchical,
                        LayoutHierarchical,
                    >::new(&mut self.graph));

                    // Show some textual details about the genome
                    ui.label(format!("Nodes: {}", self.genome.nodes.len()));
                    ui.label(format!("Connections: {}", self.genome.connections.len()));
                });
            });
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let config = NeatConfig::builder()
        .network_type(NetworkType::Feedforward)
        .activation_functions(vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Relu,
        ])
        .default_activation_function(ActivationFunction::Sigmoid)
        .input_activation_function(ActivationFunction::Tanh)
        .output_activation_function(ActivationFunction::Relu)
        .build();

    // For cartpole simulation, the network expects 4 inputs and 1 output.
    let environment = Environment::new(4, 1);
    let mut population = Population::new(config, environment)
        .with_rng(42)
        .initialize(None);

    for _ in 0..10 {
        population.evaluate_parallel(|genome| cartpole_test(genome));
        population.evolve();
        println!(
            "Generation {} - Best fitness: {}   Species count: {}",
            population.generation,
            population.best_fitness,
            population.species.len()
        );
    }

    if let Some(best) = population.get_best_genome() {
        println!("Best Genome: {:#?}", best);
        let fitness = cartpole_test(best);
        println!("Best Genome Fitness: {}", fitness);

        let app = SimulationApp::new(best.clone(), FeedforwardNetwork::new(&best).unwrap());
        return eframe::run_native(
            "Cartpole Simulation",
            eframe::NativeOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        );
    }
    Ok(())
}
