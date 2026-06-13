use libp2p::{
    gossipsub, mdns, noise, swarm::NetworkBehaviour, swarm::SwarmEvent, tcp, yamux,
    futures::StreamExt,
};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tokio::{io, io::AsyncBufReadExt, select};

#[derive(Serialize, Deserialize, Debug)]
enum SwarmMessage {
    /// A node completed tuning and is broadcasting the mathematical LoRA delta weights.
    LoRAWeightUpdate { node_id: String, checksum: String },
    /// A massive task fragmented by the CEO node sent to the Swarm.
    OrchestratedTask { task_id: String, instruction: String, payload: String },
    /// A worker node completed a sub-task and is returning the payload.
    TaskResult { task_id: String, worker_id: String, result: String },
    /// A scraping node found a massive anomaly and is tasking an inference node to evaluate it.
    InferenceTask { target_ticker: String, payload_size: usize },
    /// A node successfully evaluated a signal and is broadcasting the conclusion.
    AlphaSignal {
        ticker: String,
        confidence: f32,
        action: String,
    },
    
    // --- DISTRIBUTED CONSENSUS (RAFT-LITE) ---
    /// The active CEO node broadcasting a 5-second liveness ping.
    CeoHeartbeat { ceo_id: String, term: u64 },
    /// Replicates the CEO's active task ledger across all nodes so a Shadow CEO can seamlessly take over.
    LedgerSync { state_hash: String, active_tasks: usize },
}

/// Tracks pending tasks for fault-tolerance
#[derive(Debug, Clone)]
struct PendingTask {
    message: SwarmMessage,
    assigned_worker: Option<String>,
    dispatched_at: std::time::Instant,
    timeout_secs: u64,
}

/// The CEO Node logic that fragments tasks and handles network faults
struct SwarmOrchestrator {
    active_tasks: std::collections::HashMap<String, PendingTask>,
}

impl SwarmOrchestrator {
    fn new() -> Self {
        Self { active_tasks: std::collections::HashMap::new() }
    }

    /// Fragment a massive goal into sub-tasks for the Swarm
    fn fragment_directive(&mut self, directive: &str) -> Vec<SwarmMessage> {
        println!("[CEO] Fragmenting Directive: '{}'", directive);
        
        let sub_tasks = vec![
            SwarmMessage::OrchestratedTask {
                task_id: "T-01".to_string(),
                instruction: "Scrape SEC Filings for target".to_string(),
                payload: "https://sec.gov".to_string(),
            },
            SwarmMessage::OrchestratedTask {
                task_id: "T-02".to_string(),
                instruction: "Run local LoRA Sentiment Analysis on scraped text".to_string(),
                payload: "WAITING_ON_T-01".to_string(),
            },
            SwarmMessage::OrchestratedTask {
                task_id: "T-03".to_string(),
                instruction: "Calculate NAS100 Systemic Risk Exposure".to_string(),
                payload: "WAITING_ON_T-02".to_string(),
            }
        ];

        for task in &sub_tasks {
            if let SwarmMessage::OrchestratedTask { task_id, .. } = task {
                self.active_tasks.insert(task_id.clone(), PendingTask {
                    message: task.clone(),
                    assigned_worker: None,
                    dispatched_at: std::time::Instant::now(),
                    timeout_secs: 45, // 45 seconds before assuming the node thermal throttled or died
                });
            }
        }
        sub_tasks
    }

    /// Scan for dead nodes and return tasks that need to be aggressively reassigned
    fn check_timeouts(&mut self) -> Vec<SwarmMessage> {
        let mut to_reassign = Vec::new();
        for (id, task) in self.active_tasks.iter_mut() {
            if task.dispatched_at.elapsed().as_secs() > task.timeout_secs {
                println!("[CEO-FAULT] Task {} timed out! Worker {:?} likely dropped. Aggressive reassignment triggered.", id, task.assigned_worker);
                task.dispatched_at = std::time::Instant::now();
                task.assigned_worker = None;
                to_reassign.push(task.message.clone());
            }
        }
        to_reassign
    }
}

#[derive(NetworkBehaviour)]
struct AegisMeshBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[*] Initializing Aegis Distributed Intelligence Mesh...");
    
    // Generate a secure P2P keypair
    let local_key = libp2p::identity::Keypair::generate_ed25519();
    let local_peer_id = libp2p::PeerId::from(local_key.public());
    println!("[+] Node Identity Generated: {local_peer_id}");

    // Create the Gossipsub network behavior (How models talk to each other)
    let message_id_fn = |message: &gossipsub::Message| {
        let mut s = DefaultHasher::new();
        message.data.hash(&mut s);
        gossipsub::MessageId::from(s.finish().to_string())
    };

    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        .build()
        .expect("Failed to build gossipsub config");

    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(local_key.clone()),
        gossipsub_config,
    ).expect("Failed to build gossipsub behaviour");

    // The universal topic for the Aegis Mesh
    let global_topic = gossipsub::IdentTopic::new("aegis_global_mesh");
    gossipsub.subscribe(&global_topic)?;

    // Set up mDNS to automatically discover other MacBooks/laptops on the local network
    let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;
    let behaviour = AegisMeshBehaviour { gossipsub, mdns };

    // Build the Swarm using the v0.56 Builder Pattern
    let mut swarm = libp2p::SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|_| behaviour)?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on all interfaces on a random port
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    println!("[*] Aegis Mesh Node Online. Searching for other peer models...");

    // Setup an async reader for local terminal commands
    let mut stdin = io::BufReader::new(io::stdin()).lines();

    let mut orchestrator = SwarmOrchestrator::new();

    loop {
        select! {
            // Read terminal input to broadcast manual commands to the swarm
            Ok(Some(line)) = stdin.next_line() => {
                if line.starts_with("orchestrate") {
                    let parts: Vec<&str> = line.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        let directive = parts[1];
                        let tasks = orchestrator.fragment_directive(directive);
                        
                        for task in tasks {
                            let json = serde_json::to_vec(&task).unwrap();
                            if let Err(e) = swarm.behaviour_mut().gossipsub.publish(global_topic.clone(), json) {
                                println!("[!] Failed to broadcast task to swarm: {e:?}");
                            } else {
                                println!("[+] Task Dispatched to Mesh: {:?}", task);
                            }
                        }
                    }
                } else if line.starts_with("signal") {
                    let msg = SwarmMessage::AlphaSignal {
                        ticker: "NAS100".to_string(),
                        confidence: 0.98,
                        action: "SHORT".to_string(),
                    };
                    let json = serde_json::to_vec(&msg).unwrap();
                    if let Err(e) = swarm.behaviour_mut().gossipsub.publish(global_topic.clone(), json) {
                        println!("[!] Failed to broadcast signal: {e:?}");
                    } else {
                        println!("[+] Signal successfully broadcast to the global mesh.");
                    }
                }
            }

            // Handle incoming Swarm events (Discovery and Messages)
            event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(AegisMeshBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                    for (peer_id, multiaddr) in list {
                        println!("[+] Discovered new Aegis Node: {peer_id} at {multiaddr}");
                        swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    }
                },
                SwarmEvent::Behaviour(AegisMeshBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                    for (peer_id, _) in list {
                        println!("[-] Aegis Node disconnected: {peer_id}");
                        swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                    }
                },
                SwarmEvent::Behaviour(AegisMeshBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source: peer_id,
                    message_id: id,
                    message,
                })) => {
                    if let Ok(swarm_msg) = serde_json::from_slice::<SwarmMessage>(&message.data) {
                        println!("[Incoming] Message from {peer_id} | ID: {id}");
                        match swarm_msg {
                            SwarmMessage::LoRAWeightUpdate { node_id, checksum } => {
                                println!("    -> [WEIGHT SYNC] Node {node_id} is distributing new LoRA weights (checksum: {checksum}).");
                            },
                            SwarmMessage::OrchestratedTask { task_id, instruction, payload } => {
                                println!("    -> [SWARM DELEGATION] Received Task {task_id}: {instruction}");
                                println!("       [*] Booting local 1.58-bit engine to execute payload...");
                                
                                // Mock the worker node completing the inference
                                let result_msg = SwarmMessage::TaskResult {
                                    task_id: task_id.clone(),
                                    worker_id: local_peer_id.to_string(),
                                    result: "INFERENCE_COMPLETE_ALPHA_EXTRACTED".to_string(),
                                };
                                let json = serde_json::to_vec(&result_msg).unwrap();
                                // Note: In a real swarm, you wait for the inference thread to finish before broadcasting
                                let _ = swarm.behaviour_mut().gossipsub.publish(global_topic.clone(), json);
                            },
                            SwarmMessage::TaskResult { task_id, worker_id, result } => {
                                println!("    -> [CEO SYNTHESIS] Worker Node {worker_id} completed task {task_id}. Result: {result}");
                                // Resolve the task so it doesn't trigger a timeout reassignment
                                orchestrator.active_tasks.remove(&task_id);
                            },
                            SwarmMessage::InferenceTask { target_ticker, payload_size } => {
                                println!("    -> [TASK DELEGATED] Scraper requested inference on {target_ticker} ({payload_size} bytes). Running matrices...");
                            },
                            SwarmMessage::AlphaSignal { ticker, confidence, action } => {
                                println!("    -> [ALPHA EXTRACTED] {action} {ticker} with {confidence} confidence! Piping to Kessler...");
                            },
                            SwarmMessage::CeoHeartbeat { ceo_id, term } => {
                                // Silent heartbeat tracking, but we log the first discovery
                                // println!("    -> [RAFT-LITE] Received CEO Heartbeat from {} (Term {})", ceo_id, term);
                            },
                            SwarmMessage::LedgerSync { state_hash, active_tasks } => {
                                println!("    -> [SHADOW LEDGER] Synced active tasks ({active_tasks}) with CEO hash: {state_hash}");
                            }
                        }
                    }
                },
                SwarmEvent::NewListenAddr { address, .. } => {
                    println!("[+] Local Node listening on {address}");
                }
                _ => {}
            }
        }
    }
}
