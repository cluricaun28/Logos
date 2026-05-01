
import torch
from agent.context_engine import SemanticVectorEngine

def debug_semantic_pruning():
    print("Debugging Integration Test v3...")
    engine = SemanticVectorEngine()
    
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Let's talk about Blackwell GPUs for the server."}, # Turn 1 (H)
        {"role": "assistant", "content": "Blackwell is great for VRAM."},            # Turn 2 (H)
        {"role": "user", "content": "The power draw on these rigs is huge."},       # Turn 3 (H)
        {"role": "user", "content": "Now let's switch to DPO training weights."},     # Turn 4 (D)
        {"role": "assistant", "content": "DPO helps with epistemic truth."},         # Turn 5 (D)
        {"role": "user", "content": "I love how the weights are updated."},         # Turn 6 (D)
        {"role": "assistant", "content": "Yes, it is very efficient."},              # Turn 7 (H - transition/return)
        {"role": "user", "content": "Actually, back to the server. How much power does that box use?"}, # Turn 8 (H)
        {"role": "assistant", "content": "It uses a lot of power."},                  # Turn 9 (H)
        {"role": "user", "content": "What about cooling for the rig?"},              # Turn 10 (H)
    ]
    
    pruned = engine.archive(messages)
    
    print("\n--- PRUNED MESSAGES ---")
    for i, m in enumerate(pruned):
        print(f"{i}: {m['role']} | {m['content']}")
    
    print("\n--- VECTOR STATE ---")
    for vid, vec in engine.vectors.items():
        print(f"Vector {vid}: status={vec.status}, turns={vec.turns}")
    print(f"Current Vector: {engine.current_vector_id}")

if __name__ == "__main__":
    debug_semantic_pruning()
