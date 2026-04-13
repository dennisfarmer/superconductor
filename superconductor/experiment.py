import time
from magenta_client import MagentaClient   # ← change this to your actual filename

# ---------- Setup ----------
client = MagentaClient()
client.start()

# ---------- Wait for connection ----------
print("Waiting for connection...")
time.sleep(3)

# ---------- Phase 1: Let buffer fill ----------
print("\n=== Phase 1: Filling buffer with JAZZ ===")
print("Listen to baseline music...")

# wait ~20–25 seconds = ~10 chunks
time.sleep(25)

# ---------- Phase 2: Switch style ----------
print("\n=== Phase 2: SWITCH STYLE → ROCK ===")
print("Dropping buffer + updating recipe... LISTEN CAREFULLY")

client.update_recipe({
    "Rock": 1.0
})

# ---------- Phase 3: Observe ----------
print("\n=== Phase 3: Listening after transition ===")
print("Pay attention to continuity (chords, rhythm, abruptness)")

time.sleep(20)

# ---------- Phase 4: Another switch (optional) ----------
print("\n=== Phase 4: SWITCH AGAIN → AMBIENT ===")

client.update_recipe({
    "Ambient": 1.0
})

time.sleep(20)

print("\n=== Experiment Done ===")