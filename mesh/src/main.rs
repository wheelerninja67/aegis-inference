use libp2p::{
    gossipsub, mdns, noise, swarm::NetworkBehaviour, swarm::SwarmEvent, tcp, yamux, Multiaddr,
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
    /// A scraping node found a massive anomaly and is tasking an inference node to evaluate it.
    InferenceTask { target_ticker: String, payload_size: usize },
    /// A node successfully evaluated a signal and is broadcasting the conclusion.
    AlphaSignal {
        ticker: String,
        confidence: f32,
        action: String,
    },
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

    // Set up the TCP transport with Noise encryption and Yamux multiplexing
    let transport = libp2p::tokio_development_transport(local_key.clone())?;

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
        gossipsub::MessageAuthenticity::Signed(local_key),
        gossipsub_config,
    ).expect("Failed to build gossipsub behaviour");

    // The universal topic for the Aegis Mesh
    let global_topic = gossipsub::IdentTopic::new("aegis_global_mesh");
    gossipsub.subscribe(&global_topic)?;

    // Set up mDNS to automatically discover other MacBooks/laptops on the local network
    let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;
    let behaviour = AegisMeshBehaviour { gossipsub, mdns };

    // Build the Swarm
    let mut swarm = libp2p::SwarmBuilder::with_tokio_executor(transport, behaviour, local_peer_id).build();

    // Listen on all interfaces on a random port
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    println!("[*] Aegis Mesh Node Online. Searching for other peer models...");

    // Setup an async reader for local terminal commands
    let mut stdin = io::BufReader::new(io::stdin()).lines();

    loop {
        select! {
            // Read terminal input to broadcast manual commands to the swarm
            Ok(Some(line)) = stdin.next_line() => {
                if line.starts_with("signal") {
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
                            SwarmMessage::InferenceTask { target_ticker, payload_size } => {
                                println!("    -> [TASK DELEGATED] Scraper requested inference on {target_ticker} ({payload_size} bytes). Running matrices...");
                            },
                            SwarmMessage::AlphaSignal { ticker, confidence, action } => {
                                println!("    -> [ALPHA EXTRACTED] {action} {ticker} with {confidence} confidence! Piping to Kessler...");
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
