
import torch
from agent.context_engine import SemanticVectorEngine

def test_semantic_pruning():
    print("Starting Integration Test...")
    engine = SemanticVectorEngine()
    
    # Simulated history: System + 10 turns
    # Turns 1-3: Hardware, Turns 4-6: DPO, Turns 7-10: Hardware again
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
    
    print(f"Original length: {len(messages)}")
    
    # Trigger archive
    pruned = engine.archive(messages)
    
    print(f"Pruned length: {len(pruned)}")
    
    # Verification 1: System prompt preserved
    assert pruned[0]["role"] == "system" and "You are Hermes" in pruned[0]["content"]
    
    # Verification 2: State Map Header exists
    assert any("Conversation State" in m.get("content", "") for m in pruned if m["role"] == "system")
    print("✓ State Map Header generated.")
    
    # Verification 3: Hardware turns preserved, DPO turns evicted (except recency)
    # Turn 1-3 should be kept because they match the active vector (Hardware)
    # Turns 4-6 should be gone.
    # Turns 8-10 are protected by recency.
    
    content_blob = " ".join([m.get("content", "").lower() for m in pruned])
    assert "blackwell" in content_blob, "Failed to preserve active vector (Hardware)"
    assert "dpo" not in content_blob or "weights" not in content_blob, "Failed to evict dormant vector (DPO)"
    print("✓ Semantic pruning logic verified.")

    # Verification 4: CUDA consistency
    # Check if any tensors are on CPU while model is on CUDA
    for vid, vec in engine.vectors.items():
        if vec.embedding is not None:
            assert vec.embedding.device.type == engine.device, f"Tensor for {vid} is on wrong device!"
    print("✓ CUDA tensor consistency verified.")

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_semantic_pruning()
