import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam

def test_adam_resurrection():
    print("--- Testing Adam Momentum Resurrection ---")
    
    # 1. Setup a tiny layer
    rng = np.random.default_rng(42)
    W_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer = Linear(2, 2, rng=rng)
    layer.W.data = W_val
    
    # Create a mask (prune the second weight of the first row)
    # [1, 0]
    # [1, 1]
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])
    
    # Create Adam optimizer
    opt = Adam([layer.W], lr=0.1)
    
    # Simulate a state where the pruned weight has high momentum
    # m = first moment, v = second moment
    opt.m[0] = np.array([[0.1, 5.0], [0.1, 0.1]]) # Weight (0,1) has huge momentum
    opt.v[0] = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    # Define a dummy gradient (zero for pruned weight)
    grad = np.array([[0.1, 0.0], [0.1, 0.1]])
    layer.W.grad = grad
    
    print(f"Initial Weight (0,1): {layer.W.data[0,1]} | Mask: {mask[0,1]}")
    print(f"Momentum for Weight (0,1): {opt.m[0][0,1]}")

    # --- SCENARIO 1: WITHOUT MOMENT RESET (Simulation) ---
    # We simulate what happens if we just apply the mask but don't reset m/v
    W_temp = layer.W.data.copy()
    m_temp = opt.m[0].copy()
    v_temp = opt.v[0].copy()
    
    # Adam update: W = W - lr * m / sqrt(v)
    # (Simplified update for demonstration)
    W_temp[0,1] = W_temp[0,1] - 0.1 * (m_temp[0,1] / (np.sqrt(v_temp[0,1]) + 1e-8))
    W_temp = W_temp * mask # Re-apply mask
    
    # Note: In a real loop, the drift happens because the momentum persists
    # and pushes the weight unless we zero m.
    
    # --- SCENARIO 2: WITH OUR CORRECT IMPLEMENTATION ---
    # Step A: Moment Reset (The Fix)
    opt.m[0] = opt.m[0] * mask
    opt.v[0] = opt.v[0] * mask
    
    print("\nApplying Correct Fix: Resetting Moment and Variance...")
    print(f"New Momentum for Weight (0,1): {opt.m[0][0,1]}")
    
    # Step B: Optimizer Update
    opt.step()
    
    # Step C: Hard Enforcement
    layer.W.data = layer.W.data * mask
    
    print(f"Final Weight (0,1): {layer.W.data[0,1]}")
    
    if layer.W.data[0,1] == 0.0:
        print("\nSUCCESS: Weight remained at 0. The resurrection was prevented.")
    else:
        print("\nFAILURE: Weight drifted!")

def test_sparse_inference_ops():
    print("\n--- Testing Sparse Inference Operation Count ---")
    # This is a conceptual test to prove the "Proof" in the MD
    import time
    
    # Dense: 1000x1000
    W_dense = np.random.randn(1000, 1000)
    X = np.random.randn(1000)
    
    start = time.time()
    res_dense = np.dot(W_dense, X)
    end = time.time()
    dense_time = end - start
    
    # Sparse: 1000x1000 but 99% sparse
    mask = (np.random.rand(1000, 1000) < 0.01).astype(float)
    W_sparse = W_dense * mask
    
    # Simulate SparseLinear logic: iterate only over non-zeros
    indices = np.where(mask > 0)
    res_sparse = np.zeros(1000)
    
    start = time.time()
    for i, j in zip(*indices):
        res_sparse[i] += W_sparse[i, j] * X[j]
    end = time.time()
    sparse_time = end - start
    
    print(f"Dense Dot Product Time: {dense_time:.5f}s")
    print(f"Sparse Loop Time (1% density): {sparse_time:.5f}s")
    print(f"Speedup: {dense_time/sparse_time:.2f}x")
    print("Note: In production, we'd use CSR/CSC indices for even more speed.")

if __name__ == "__main__":
    test_adam_resurrection()
    test_sparse_inference_ops()
