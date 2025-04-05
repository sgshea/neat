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
use std::f32::consts::PI;
use std::time::{Duration, Instant};

/// The pendulum starts hanging down (under the cart) and the neural network
/// must learn to swing it up and balance it in the inverted position.
/// The network takes 4 inputs: cart position, cart velocity, pendulum angle, and pendulum angular velocity.
/// It outputs a force direction to apply to the cart.
fn inverted_pendulum_test(genome: &Genome) -> f32 {
    let mut nn = FeedforwardNetwork::new(genome).unwrap();

    // Simulation parameters
    let dt = 0.02; // seconds
    let gravity = 9.8;
    let mass_cart = 1.0;
    let mass_pendulum = 0.1;
    let pendulum_length = 0.5; // half-length of pendulum
    let force_mag = 10.0;
    let max_steps = 500; // Increased to allow time for swing-up

    // Initial state - pendulum starts hanging down (PI radians)
    let (mut x, mut x_dot, mut theta, mut theta_dot) = (0.0, 0.0, PI, 0.0);

    let mut fitness = 0.0;
    let mut upright_steps = 0;

    for _ in 0..max_steps {
        // Get neural network output based on current state
        let inputs = vec![x, x_dot, theta, theta_dot];
        let output = nn.activate(&inputs).unwrap();
        let force = if output[0] > 0.5 {
            force_mag
        } else {
            -force_mag
        };

        // Calculate dynamics using inverted pendulum equations of motion
        let costheta = theta.cos();
        let sintheta = theta.sin();

        // Calculate acceleration of the pendulum angle
        let temp = (force + mass_pendulum * pendulum_length * theta_dot.powi(2) * sintheta)
            / (mass_cart + mass_pendulum);
        let theta_acc = (gravity * sintheta - costheta * temp)
            / (pendulum_length
                * (4.0 / 3.0 - mass_pendulum * costheta.powi(2) / (mass_cart + mass_pendulum)));

        // Calculate acceleration of the cart
        let x_acc = temp
            - mass_pendulum * pendulum_length * theta_acc * costheta / (mass_cart + mass_pendulum);

        // Update state using Euler integration
        x += dt * x_dot;
        x_dot += dt * x_acc;
        theta += dt * theta_dot;
        theta_dot += dt * theta_acc;

        // Normalize theta to range [-PI, PI]
        theta = ((theta + PI) % (2.0 * PI)) - PI;

        // Check if pendulum is upright (near 0 radians)
        let upright_threshold = 0.2; // About 11.5 degrees from vertical
        if theta.abs() < upright_threshold {
            upright_steps += 1;
            // Provide strong reward for balancing upright
            fitness += 1.0;
        } else {
            // Provide small reward for getting closer to upright position
            // This encourages the swing-up behavior
            fitness += 0.1 * (1.0 - (theta.abs() / PI));
        }

        // Failure conditions: cart moves too far
        if x.abs() > 2.5 {
            break;
        }
    }

    // Final fitness heavily weights time spent balanced upright
    // but also rewards progress toward swing-up
    fitness + (upright_steps as f32) * 2.0
}

/// This EGUI application displays the real-time inverted pendulum simulation (left pane)
/// and a network visualization of the best genome (right pane).
struct InvertedPendulumApp<'n> {
    // Genome and network derived from the best genome
    genome: Genome,
    network: FeedforwardNetwork<'n>,
    graph: egui_graphs::Graph,
    // Pendulum simulation state
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    // Physics parameters
    dt: f32,
    gravity: f32,
    mass_cart: f32,
    mass_pendulum: f32,
    pendulum_length: f32,
    force_mag: f32,
    last_update: Instant,
    // Track performance
    upright_steps: u32,
    max_upright_steps: u32,
    total_steps: u32,
    swing_up_complete: bool,
}

impl<'n> InvertedPendulumApp<'n> {
    fn new(genome: Genome, network: FeedforwardNetwork<'n>) -> Self {
        let graph = egui_graphs::Graph::from(&generate_graph(&genome));
        InvertedPendulumApp {
            genome,
            graph,
            network,
            x: 0.0,
            x_dot: 0.0,
            theta: PI, // Start hanging down
            theta_dot: 0.0,
            dt: 0.02,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pendulum: 0.1,
            pendulum_length: 0.5,
            force_mag: 10.0,
            last_update: Instant::now(),
            upright_steps: 0,
            max_upright_steps: 0,
            total_steps: 0,
            swing_up_complete: false,
        }
    }

    /// Update the inverted pendulum simulation
    fn update_simulation(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_update) < Duration::from_secs_f32(self.dt) {
            return;
        }
        self.last_update = now;
        self.total_steps += 1;

        // Get control action from neural network
        let inputs = vec![self.x, self.x_dot, self.theta, self.theta_dot];
        let output = self.network.activate(&inputs).unwrap();
        let force = if output[0] > 0.5 {
            self.force_mag
        } else {
            -self.force_mag
        };

        // Calculate dynamics
        let costheta = self.theta.cos();
        let sintheta = self.theta.sin();
        let temp = (force
            + self.mass_pendulum * self.pendulum_length * self.theta_dot.powi(2) * sintheta)
            / (self.mass_cart + self.mass_pendulum);
        let theta_acc = (self.gravity * sintheta - costheta * temp)
            / (self.pendulum_length
                * (4.0 / 3.0
                    - self.mass_pendulum * costheta.powi(2)
                        / (self.mass_cart + self.mass_pendulum)));
        let x_acc = temp
            - self.mass_pendulum * self.pendulum_length * theta_acc * costheta
                / (self.mass_cart + self.mass_pendulum);

        // Update state
        self.x += self.dt * self.x_dot;
        self.x_dot += self.dt * x_acc;
        self.theta += self.dt * self.theta_dot;
        self.theta_dot += self.dt * theta_acc;

        // Normalize theta to range [-PI, PI]
        self.theta = ((self.theta + PI) % (2.0 * PI)) - PI;

        // Check if pendulum is upright
        let upright_threshold = 0.2;
        if self.theta.abs() < upright_threshold {
            self.upright_steps += 1;
            // Mark as successful swing-up once it is balanced for some time
            if self.upright_steps > 50 && !self.swing_up_complete {
                self.swing_up_complete = true;
            }
        }

        self.max_upright_steps = self.max_upright_steps.max(self.upright_steps);

        // Reset if cart moves too far or after a long run
        if self.x.abs() > 2.5 || self.total_steps > 1000 {
            self.x = 0.0;
            self.x_dot = 0.0;
            self.theta = PI; // Reset to hanging down
            self.theta_dot = 0.0;
            self.upright_steps = 0;
            self.total_steps = 0;
            self.swing_up_complete = false;
        }
    }

    /// Render the inverted pendulum simulation
    fn draw_simulation(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();

        // Draw track
        let track_y = rect.center().y + 50.0;
        let track_height = 5.0;
        let track_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left(), track_y - track_height / 2.0),
            egui::vec2(rect.width(), track_height),
        );
        painter.rect_filled(track_rect, 0.0, egui::Color32::from_rgb(100, 100, 100));

        // Draw boundaries
        let boundary_left = rect.center().x - 2.5 * rect.width() / 6.0;
        let boundary_right = rect.center().x + 2.5 * rect.width() / 6.0;
        let boundary_height = 100.0;
        painter.line_segment(
            [
                egui::pos2(boundary_left, track_y - boundary_height),
                egui::pos2(boundary_left, track_y + 10.0),
            ],
            egui::Stroke::new(2.0, egui::Color32::RED),
        );
        painter.line_segment(
            [
                egui::pos2(boundary_right, track_y - boundary_height),
                egui::pos2(boundary_right, track_y + 10.0),
            ],
            egui::Stroke::new(2.0, egui::Color32::RED),
        );

        // Draw cart
        let scale = rect.width() / 6.0;
        let cart_w = 50.0;
        let cart_h = 30.0;
        let sim_to_screen_x = |x: f32| rect.center().x + x * scale;
        let cart_x = sim_to_screen_x(self.x) - cart_w / 2.0;
        let cart_rect = egui::Rect::from_min_size(
            egui::pos2(cart_x, track_y - cart_h),
            egui::vec2(cart_w, cart_h),
        );

        // Change cart color based on swing-up status
        let cart_color = if self.swing_up_complete {
            egui::Color32::from_rgb(50, 200, 50) // Green when successful
        } else if self.upright_steps > 0 {
            egui::Color32::from_rgb(50, 150, 200) // Blue when making progress
        } else {
            egui::Color32::DARK_BLUE // Default color
        };

        painter.rect_filled(cart_rect, 4.0, cart_color);

        // Draw the pendulum
        let cart_center = egui::pos2(sim_to_screen_x(self.x), track_y - cart_h / 2.0);
        let pendulum_length_px = self.pendulum_length * scale * 2.0; // scale for visibility
        let pendulum_end = egui::pos2(
            cart_center.x + pendulum_length_px * self.theta.sin(),
            cart_center.y - pendulum_length_px * self.theta.cos(),
        );
        painter.line_segment(
            [cart_center, pendulum_end],
            egui::Stroke {
                width: 4.0,
                color: egui::Color32::from_rgb(200, 50, 50),
            },
        );
        painter.circle_filled(pendulum_end, 8.0, egui::Color32::from_rgb(200, 50, 50));

        // Draw target position (upright)
        let target_end = egui::pos2(cart_center.x, cart_center.y - pendulum_length_px);
        painter.circle_stroke(
            target_end,
            10.0,
            egui::Stroke::new(1.0, egui::Color32::YELLOW),
        );
    }
}

impl App for InvertedPendulumApp<'_> {
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

    // For inverted pendulum simulation, the network expects 4 inputs and 1 output
    let environment = Environment::new(4, 1);
    let mut population = Population::new(config, environment)
        .with_rng(42)
        .initialize(None);

    for _ in 0..150 {
        population.evaluate_parallel(|genome| inverted_pendulum_test(genome));
        population.evolve();
        println!(
            "Generation {} - Best fitness: {}   Species count: {}",
            population.generation,
            population.best_fitness,
            population.species.len()
        );
    }

    if let Some(best) = population.get_best_genome() {
        let fitness = inverted_pendulum_test(best);
        println!("Swing-Up Performance: {}", fitness);

        // Launch the EGUI visualization
        let app = InvertedPendulumApp::new(best.clone(), FeedforwardNetwork::new(&best).unwrap());
        let native_options = eframe::NativeOptions {
            ..Default::default()
        };
        return eframe::run_native(
            "Inverted Pendulum",
            native_options,
            Box::new(|_cc| Ok(Box::new(app))),
        );
    }
    Ok(())
}
