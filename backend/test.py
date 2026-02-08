import time
import os
import json
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair as SolanaKeypair
from solders.pubkey import Pubkey

# Solana config
SOLANA_RPC_URL = "https://api.devnet.solana.com"
SOLANA_ENABLED = os.environ.get("SOLANA_ENABLED", "true").lower() == "true"

# Global Wallet
authority_keypair = None 
solana_client = None

def setup_solana_wallet():
    """Initialize the global wallet and fund it if necessary."""
    global authority_keypair, solana_client
    
    if not SOLANA_ENABLED:
        print("Solana integration disabled via config.")
        return

    try:
        solana_client = SolanaClient(SOLANA_RPC_URL)
        
        # --- FIXED LOADING LOGIC ---
        if os.path.exists("authority_keypair.json"):
            with open("authority_keypair.json", "r") as f:
                raw = f.read()
                # Load the list of integers from JSON
                key_list = json.loads(raw)
                # Convert list back to bytes for the Keypair constructor
                authority_keypair = SolanaKeypair.from_bytes(bytes(key_list))
                print(f"Loaded existing Solana Wallet: {authority_keypair.pubkey()}")
        else:
            # --- FIXED SAVING LOGIC ---
            authority_keypair = SolanaKeypair()
            # Convert keypair to bytes, then to a standard Python list for JSON
            key_as_list = list(bytes(authority_keypair))
            
            with open("authority_keypair.json", "w") as f:
                f.write(json.dumps(key_as_list))
            print(f"Generated NEW Solana Wallet: {authority_keypair.pubkey()}")

        # Check Balance
        print("Checking balance...")
        balance_resp = solana_client.get_balance(authority_keypair.pubkey())
        lamports = balance_resp.value
        
        # If low balance (< 0.5 SOL), request airdrop
        if lamports < 500_000_000:
            print(f"Balance low ({lamports} lamports). Requesting Devnet Airdrop...")
            try:
                # 1 SOL = 1,000,000,000 Lamports
                solana_client.request_airdrop(authority_keypair.pubkey(), 1_000_000_000)
                time.sleep(2) 
                print("Airdrop requested successfully.")
            except Exception as e:
                print(f"Airdrop failed (might be rate limited): {e}")
        else:
            print(f"Wallet funded: {lamports / 1_000_000_000:.2f} SOL")

    except Exception as e:
        print(f"Failed to initialize Solana: {e}")

if __name__ == "__main__":
    print("--- Starting Test ---")
    setup_solana_wallet()
    print("--- Test Complete ---")