"""
GPU-Accelerated Hamiltonian Solver Module

This module provides GPU-accelerated solvers for finding ground states
and solving constraint equations in quantum gravity calculations.
"""

import numpy as np
from typing import Optional

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def solve_constraint_gpu(hamiltonian_matrix: np.ndarray,
                         initial_state: np.ndarray,
                         num_steps: int = 1000,
                         lr: float = 1e-3) -> np.ndarray:
    """
    Example gradient-based solver using PyTorch to find the kernel of Ĥ (Ĥ|ψ⟩ = 0).
    
    Args:
        hamiltonian_matrix: CPU numpy array (Hermitian)
        initial_state: CPU numpy array (complex), shape (dim, 1)
        num_steps: Number of optimization steps
        lr: Learning rate for optimizer
        
    Returns:
        Normalized ground state as numpy array
        
    Raises:
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU-accelerated solver.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = torch.tensor(hamiltonian_matrix, dtype=torch.cdouble, device=device)
    psi = torch.tensor(initial_state, dtype=torch.cdouble, device=device).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([psi], lr=lr)
    for step in range(num_steps):
        # Normalize psi at each iteration
        psi_norm = psi / torch.norm(psi)
        loss = torch.matmul(psi_norm.conj().t(), H @ psi_norm).real
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"[GPU Solver] Step {step}, ⟨ψ|Ĥ|ψ⟩ = {loss.item():.3e}")
            if abs(loss.item()) < 1e-10:
                break

    return psi_norm.detach().cpu().numpy()


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


def get_device_info() -> dict:
    """Get information about available compute devices."""
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "device_names": []
    }
    
    if TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i) 
                                  for i in range(torch.cuda.device_count())]
    
    return info


def solve_eigenvalue_problem_gpu(matrix: np.ndarray, 
                                k: int = 10,
                                which: str = 'smallest') -> tuple:
    """
    Solve eigenvalue problem using GPU acceleration.
    
    Args:
        matrix: Hermitian matrix as numpy array
        k: Number of eigenvalues/eigenvectors to compute
        which: Which eigenvalues to find ('smallest', 'largest')
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) as numpy arrays
        
    Raises:
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GPU-accelerated eigenvalue solver.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor(matrix, dtype=torch.cdouble, device=device)
    
    # Use PyTorch's eigenvalue decomposition
    eigenvals, eigenvecs = torch.linalg.eigh(A)
    
    # Sort and select eigenvalues
    if which == 'smallest':
        indices = torch.argsort(eigenvals.real)[:k]
    elif which == 'largest':
        indices = torch.argsort(eigenvals.real, descending=True)[:k]
    else:
        raise ValueError("which must be 'smallest' or 'largest'")
    
    selected_vals = eigenvals[indices]
    selected_vecs = eigenvecs[:, indices]
    
    return selected_vals.cpu().numpy(), selected_vecs.cpu().numpy()


class GPUConstraintSolver:
    """
    Advanced GPU-accelerated constraint solver for quantum gravity.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the GPU constraint solver.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU constraint solver.")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initialized GPU constraint solver on device: {self.device}")
    
    def solve_wheeler_dewitt(self, 
                           hamiltonian: np.ndarray,
                           initial_guess: np.ndarray,
                           tolerance: float = 1e-10,
                           max_iterations: int = 5000) -> dict:
        """
        Solve the Wheeler-DeWitt equation Ĥ|ψ⟩ = 0.
        
        Args:
            hamiltonian: Hamiltonian matrix
            initial_guess: Initial state vector
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with solution and convergence info
        """
        H = torch.tensor(hamiltonian, dtype=torch.cdouble, device=self.device)
        psi = torch.tensor(initial_guess, dtype=torch.cdouble, device=self.device).clone()
        psi.requires_grad_(True)
        
        optimizer = torch.optim.LBFGS([psi], lr=1.0, max_iter=20)
        
        def closure():
            optimizer.zero_grad()
            psi_normalized = psi / torch.norm(psi)
            constraint_violation = torch.norm(H @ psi_normalized)**2
            constraint_violation.backward()
            return constraint_violation
        
        converged = False
        final_violation = float('inf')
        
        for iteration in range(max_iterations // 20):  # LBFGS does multiple steps per call
            loss = optimizer.step(closure)
            final_violation = loss.item()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration*20}, Constraint violation: {final_violation:.3e}")
            
            if final_violation < tolerance:
                converged = True
                break
        
        # Final normalization
        with torch.no_grad():
            psi_final = psi / torch.norm(psi)
        
        return {
            "solution": psi_final.detach().cpu().numpy(),
            "constraint_violation": final_violation,
            "converged": converged,
            "iterations": iteration * 20,
            "device": str(self.device)
        }
    
    def compute_expectation_value(self, 
                                operator: np.ndarray,
                                state: np.ndarray) -> complex:
        """
        Compute expectation value ⟨ψ|Ô|ψ⟩ on GPU.
        
        Args:
            operator: Operator matrix
            state: Quantum state vector
            
        Returns:
            Expectation value as complex number
        """
        O = torch.tensor(operator, dtype=torch.cdouble, device=self.device)
        psi = torch.tensor(state, dtype=torch.cdouble, device=self.device)
        
        psi_normalized = psi / torch.norm(psi)
        expectation = torch.vdot(psi_normalized, O @ psi_normalized)
        
        return complex(expectation.cpu().item())
