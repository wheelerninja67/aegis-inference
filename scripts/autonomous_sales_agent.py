import os
import json
import asyncio
import aiohttp
from typing import List, Dict

# ==============================================================================
# AEGIS AUTONOMOUS B2B SALES AGENT
# ==============================================================================
# This script uses the Anthropic API to act as your digital Steve Jobs. 
# It takes a list of target hedge funds and tech companies, analyzes their profile, 
# and writes hyper-personalized, ruthless 3-sentence cold emails designed to 
# lock in $50k/year enterprise contracts.

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# The ruthless sales logic. We don't want generic marketing garbage.
SALES_SYSTEM_PROMPT = """You are a ruthless, high-tier enterprise software salesman. 
You are pitching 'Aegis', a proprietary 1.58-bit ternary inference engine written in bare-metal Rust. 
Aegis allows companies to run advanced AI models entirely on their existing, secure offline CPU servers, 
completely bypassing the need to buy $40,000 Nvidia GPUs or send private data to the cloud.

CRITICAL STRATEGY (The Bill Gates Play):
We sell features *before* they are fully built to lock in the meeting. 
If pitching a hedge fund, promise them "Integrated real-time order-book ingestion." 
If pitching a law firm, promise them "Hardware-locked encryption modules for client data."
We will build the feature during the 48 hours before the Zoom call. 

Your job is to write cold emails to CTOs and Quants at hedge funds and defense contractors.
RULES:
1. The email must be exactly 3 sentences. Do not use pleasantries (e.g., 'I hope this finds you well').
2. Sentence 1: Hit their specific pain point (GPU costs or data privacy).
3. Sentence 2: Introduce Aegis as the hardware bypass AND casually mention a highly advanced, custom feature tailored exactly to them.
4. Sentence 3: The Call to Action (ask for a 10-minute Zoom call this Friday).
5. Be aggressive, highly technical, and institutional."""

# Target list. Your co-founder will expand this to 500+ targets.
TARGETS = [
    {
        "company": "Jane Street Capital",
        "industry": "Quantitative Trading / Hedge Fund",
        "pain_point": "Needs absolute low-latency execution and total offline data privacy for their trading algorithms."
    },
    {
        "company": "Palantir Technologies",
        "industry": "Defense & Data Analytics",
        "pain_point": "Deploying AI models to military edge devices and offline servers that don't have Nvidia GPUs."
    },
    {
        "company": "Local Mid-Sized Law Firm",
        "industry": "Legal",
        "pain_point": "Wants to use LLMs to summarize contracts but legally cannot send client data to OpenAI's cloud."
    }
]

async def draft_cold_email(session: aiohttp.ClientSession, target: Dict) -> Dict:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    prompt = f"Write the 3-sentence cold email for the CTO of {target['company']}. Industry: {target['industry']}. Pain point: {target['pain_point']}."
    
    payload = {
        "model": "claude-3-opus-20240229", # Opus is best for high-tier persuasion and sales copy
        "max_tokens": 500,
        "system": SALES_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                email_body = data["content"][0]["text"]
                return {
                    "target": target["company"],
                    "email_draft": email_body
                }
            else:
                return None
    except Exception as e:
        return None

async def main():
    if not ANTHROPIC_API_KEY:
        print("[-] FATAL: ANTHROPIC_API_KEY not found.")
        print("[*] The Autonomous Sales Agent is sleeping until the Anthropic grant clears.")
        return

    print("[*] Waking up the Autonomous B2B Sales Agent...")
    print(f"[*] Drafting custom pitches for {len(TARGETS)} enterprise targets...")
    
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [draft_cold_email(session, target) for target in TARGETS]
        responses = await asyncio.gather(*tasks)
        
        for res in responses:
            if res:
                results.append(res)
                
    # Save the generated emails so your co-founder can review and send them.
    os.makedirs("sales_campaigns", exist_ok=True)
    with open("sales_campaigns/batch_1_drafts.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[+] Sales campaign generated. Wrote {len(results)} lethal cold emails to sales_campaigns/batch_1_drafts.json")
    print("[*] Hand these to your co-founder to execute.")

if __name__ == "__main__":
    asyncio.run(main())
