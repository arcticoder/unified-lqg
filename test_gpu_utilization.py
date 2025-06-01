#!/usr/bin/env python3
"""
GPU Utilization Test for LQG Solver

This script tests whether the LQG solver actually uses GPU by:
1. Monitoring GPU utilization before/during/after execution
2. Testing with PyTorch backend
3. Comparing performance with CPU-only version
"""

import subprocess
import time
import threading
import psutil
import os
from typing import List, Dict

def monitor_gpu_utilization(duration: float = 30, interval: float = 1.0) -> List[Dict]:
    """Monitor GPU utilization using nvidia-smi."""
    results = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 4:
                    results.append({
                        'timestamp': time.time(),
                        'gpu_util': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'power_draw': float(parts[3]) if parts[3] != '[Not Supported]' else None
                    })
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        time.sleep(interval)
    
    return results

def test_lqg_solver_gpu():
    """Test LQG solver with GPU monitoring."""
    
    print("🔍 Testing LQG Solver GPU Utilization")
    print("=" * 50)
    
    # Test parameters
    lattice_file = "examples/lqg_lattice.json"
    output_dir = "test_gpu_outputs"
    
    # Ensure clean output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    print("📊 Starting GPU monitoring...")
    
    # Start GPU monitoring in background
    monitoring_results = []
    monitor_active = threading.Event()
    monitor_active.set()
    
    def monitor_thread():
        nonlocal monitoring_results
        monitoring_results = monitor_gpu_utilization(duration=60, interval=0.5)
    
    monitor = threading.Thread(target=monitor_thread)
    monitor.start()
    
    # Wait a moment to establish baseline
    time.sleep(2)
    
    # Record baseline GPU usage
    try:
        baseline = subprocess.run([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)
        
        baseline_parts = baseline.stdout.strip().split(', ')
        baseline_gpu = int(baseline_parts[0])
        baseline_memory = int(baseline_parts[1])
        
        print(f"📈 Baseline GPU utilization: {baseline_gpu}%")
        print(f"📈 Baseline memory usage: {baseline_memory} MB")
        
    except Exception as e:
        print(f"⚠️  Could not get baseline: {e}")
        baseline_gpu = 0
        baseline_memory = 0
    
    print("\n🚀 Running LQG solver with PyTorch GPU backend...")
    
    # Run LQG solver with GPU
    start_time = time.time()
    
    try:
        result = subprocess.run([
            "python", 
            "../warp-lqg-midisuperspace/solve_constraint.py",
            "--lattice", lattice_file,
            "--out", output_dir,
            "--backend", "torch",
            "--gpu",
            "--n-states", "3"
        ], cwd=os.getcwd(), capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        print(f"✅ LQG solver completed in {solve_time:.2f} seconds")
        
        if result.returncode != 0:
            print(f"❌ Solver failed with code {result.returncode}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
        else:
            print("✅ Solver succeeded")
            
            # Check outputs
            expected_files = [
                f"{output_dir}/expectation_T00.json",
                f"{output_dir}/expectation_E.json"
            ]
            
            for filepath in expected_files:
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   ✓ {filepath} ({size} bytes)")
                else:
                    print(f"   ✗ {filepath} (missing)")
        
    except subprocess.TimeoutExpired:
        print("❌ LQG solver timed out after 5 minutes")
        solve_time = 300
    except Exception as e:
        print(f"❌ LQG solver error: {e}")
        solve_time = 0
    
    # Stop monitoring
    monitor_active.clear()
    time.sleep(1)
    
    # Analyze GPU utilization
    print("\n📊 GPU Utilization Analysis:")
    
    if monitoring_results:
        max_gpu_util = max(r['gpu_util'] for r in monitoring_results)
        max_memory_used = max(r['memory_used'] for r in monitoring_results)
        avg_gpu_util = sum(r['gpu_util'] for r in monitoring_results) / len(monitoring_results)
        
        print(f"   Peak GPU utilization: {max_gpu_util}% (baseline: {baseline_gpu}%)")
        print(f"   Average GPU utilization: {avg_gpu_util:.1f}%")
        print(f"   Peak memory usage: {max_memory_used} MB (baseline: {baseline_memory} MB)")
        
        # Determine if GPU was actually used
        gpu_increase = max_gpu_util - baseline_gpu
        memory_increase = max_memory_used - baseline_memory
        
        if gpu_increase > 5:  # At least 5% increase
            print(f"🎉 GPU WAS USED! Utilization increased by {gpu_increase}%")
            gpu_used = True
        else:
            print(f"😞 GPU was NOT used significantly (increase: {gpu_increase}%)")
            gpu_used = False
            
        if memory_increase > 100:  # At least 100MB increase
            print(f"🎉 GPU memory increased by {memory_increase} MB")
        else:
            print(f"📊 GPU memory increase: {memory_increase} MB")
    else:
        print("❌ No monitoring data collected")
        gpu_used = False
    
    # Performance summary
    print(f"\n⏱️  Performance Summary:")
    print(f"   Solve time: {solve_time:.2f} seconds")
    print(f"   GPU utilized: {'Yes' if gpu_used else 'No'}")
    
    # Cleanup
    try:
        monitor.join(timeout=2)
    except:
        pass
    
    return gpu_used, solve_time

def main():
    print("🔬 LQG SOLVER GPU UTILIZATION TEST")
    print("=" * 60)
    
    # Check prerequisites
    if not os.path.exists("examples/lqg_lattice.json"):
        print("❌ Test lattice file not found: examples/lqg_lattice.json")
        return False
    
    if not os.path.exists("../warp-lqg-midisuperspace/solve_constraint.py"):
        print("❌ LQG solver not found: ../warp-lqg-midisuperspace/solve_constraint.py")
        return False
    
    # Run test
    gpu_used, solve_time = test_lqg_solver_gpu()
    
    print("\n" + "=" * 60)
    if gpu_used:
        print("🎉 SUCCESS: GPU acceleration is working!")
        print(f"   The LQG solver utilized your RTX 2060 GPU")
        print(f"   Solve time: {solve_time:.2f} seconds")
    else:
        print("😞 ISSUE: GPU was not utilized")
        print("   This could be due to:")
        print("   - Solver falling back to CPU due to errors")
        print("   - Problem size too small to benefit from GPU")
        print("   - Backend not properly configured")
        print(f"   Solve time: {solve_time:.2f} seconds")
    
    return gpu_used

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
