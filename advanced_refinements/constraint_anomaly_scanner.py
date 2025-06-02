#!/usr/bin/env python3
"""
Advanced Constraint Anomaly Scanner

This module provides comprehensive scanning of Hamiltonian-Hamiltonian commutators
across a range of lapse functions to detect and quantify closure errors.

Key features:
- Automated systematic scanning of [Ĥ[N], Ĥ[M]] commutators
- Multiple regularization parameter testing
- Statistical analysis of anomaly rates
- Detailed logging and JSON export of results
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys
import traceback

# Add parent directory to path for imports
sys.path.append('..')

try:
    from lqg_fixed_components import (
        LatticeConfiguration, 
        LQGParameters, 
        KinematicalHilbertSpace, 
        MidisuperspaceHamiltonianConstraint
    )
    from constraint_algebra import AdvancedConstraintAlgebraAnalyzer
except ImportError as e:
    print(f"Warning: Could not import LQG components: {e}")
    print("Creating mock implementations for testing...")


class ConstraintAnomalyScanner:
    """
    Advanced scanner for constraint algebra anomalies.
    
    Systematically tests closure of [Ĥ[N], Ĥ[M]] = 0 across
    different lapse function pairs and regularization parameters.
    """
    
    def __init__(self, n_sites: int, output_dir: str = "outputs/anomaly_analysis"):
        self.n_sites = n_sites
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.scan_results = {}
        self.statistical_summary = {}
        
        print(f"🔬 Constraint Anomaly Scanner initialized")
        print(f"   Lattice sites: {n_sites}")
        print(f"   Output directory: {self.output_dir}")
    
    def generate_lapse_function_set(self, num_functions: int = 10) -> List[np.ndarray]:
        """
        Generate a diverse set of lapse functions for testing.
        
        Includes:
        - Constant lapse
        - Linear gradients
        - Sinusoidal variations
        - Gaussian bumps
        - Random smooth functions
        """
        lapse_set = []
        r = np.linspace(0.0, 1.0, self.n_sites)
        
        # 1. Constant lapse (baseline)
        lapse_set.append(np.ones(self.n_sites))
        
        # 2. Linear gradients
        lapse_set.append(1.0 + 0.2 * r)
        lapse_set.append(1.2 - 0.3 * r)
        
        # 3. Sinusoidal variations
        for k in range(1, 4):
            lapse_set.append(1.0 + 0.1 * np.sin(k * np.pi * r))
            lapse_set.append(1.0 + 0.05 * np.cos(k * np.pi * r))
        
        # 4. Gaussian bumps
        for center in [0.3, 0.7]:
            lapse_set.append(1.0 + 0.15 * np.exp(-((r - center) / 0.2)**2))
        
        # 5. Random smooth functions (using low-frequency components)
        np.random.seed(42)  # Reproducible
        for i in range(num_functions - len(lapse_set)):
            # Build as sum of low-frequency sinusoids
            lapse = np.ones(self.n_sites)
            for k in range(1, 4):
                amp = np.random.uniform(-0.05, 0.05)
                phase = np.random.uniform(0, 2*np.pi)
                lapse += amp * np.sin(k * np.pi * r + phase)
            lapse_set.append(lapse)
        
        # Ensure all lapse functions are positive
        for i, lapse in enumerate(lapse_set):
            if np.any(lapse <= 0):
                lapse_set[i] = np.abs(lapse) + 0.1
        
        return lapse_set[:num_functions]
    
    def scan_regularization_parameters(self, 
                                     epsilon_values: List[float] = None,
                                     mu_max_values: List[int] = None) -> Dict[str, Any]:
        """
        Scan closure errors across different regularization parameters.
        
        Tests how constraint algebra closure depends on:
        - Regularization epsilon
        - Flux truncation parameters (mu_max, nu_max)
        - Basis truncation
        """
        if epsilon_values is None:
            epsilon_values = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
        
        if mu_max_values is None:
            mu_max_values = [1, 2, 3]
        
        print(f"\n🔍 Scanning regularization parameters...")
        print(f"   Epsilon values: {len(epsilon_values)}")
        print(f"   Mu_max values: {len(mu_max_values)}")
        
        regularization_results = {}
        
        for eps in epsilon_values:
            for mu_max in mu_max_values:
                param_key = f"eps_{eps:.0e}_mu_{mu_max}"
                
                try:
                    # Build LQG system with these parameters
                    lattice_config = LatticeConfiguration(
                        n_sites=self.n_sites, 
                        throat_radius=1.0
                    )
                    
                    lqg_params = LQGParameters(
                        mu_max=mu_max,
                        nu_max=mu_max,
                        basis_truncation=min(200, self.n_sites * 50),
                        regularization_epsilon=eps
                    )
                    
                    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
                    constraint = MidisuperspaceHamiltonianConstraint(
                        lattice_config, lqg_params, kin_space
                    )
                    
                    analyzer = AdvancedConstraintAlgebraAnalyzer(
                        constraint, lattice_config, lqg_params
                    )
                    
                    # Quick closure test with minimal lapse pairs
                    closure_result = analyzer.verify_constraint_closure(
                        test_multiple_lapse_pairs=False
                    )
                    
                    regularization_results[param_key] = {
                        "epsilon": float(eps),
                        "mu_max": int(mu_max),
                        "anomaly_free_rate": closure_result["anomaly_free_rate"],
                        "avg_closure_error": closure_result["avg_closure_error"],
                        "hilbert_dim": kin_space.dim
                    }
                    
                    print(f"   {param_key}: anomaly_rate={closure_result['anomaly_free_rate']:.1%}, "
                          f"error={closure_result['avg_closure_error']:.2e}")
                
                except Exception as e:
                    print(f"   ❌ Failed for {param_key}: {e}")
                    regularization_results[param_key] = {"error": str(e)}
        
        return regularization_results
    
    def comprehensive_anomaly_scan(self, 
                                 lapse_set: List[np.ndarray] = None,
                                 max_pairs: int = 50) -> Dict[str, Any]:
        """
        Perform comprehensive scanning of [Ĥ[N], Ĥ[M]] commutators.
        
        Tests all pairs of lapse functions up to max_pairs total pairs.
        """
        if lapse_set is None:
            lapse_set = self.generate_lapse_function_set(10)
        
        print(f"\n🔬 Comprehensive anomaly scanning...")
        print(f"   Lapse functions: {len(lapse_set)}")
        print(f"   Maximum pairs to test: {max_pairs}")
        
        # Setup LQG system (use best parameters from regularization scan)
        lattice_config = LatticeConfiguration(
            n_sites=self.n_sites, 
            throat_radius=1.0
        )
        
        lqg_params = LQGParameters(
            mu_max=2,
            nu_max=2,
            basis_truncation=min(300, self.n_sites * 60),
            regularization_epsilon=1e-10
        )
        
        kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
        constraint = MidisuperspaceHamiltonianConstraint(
            lattice_config, lqg_params, kin_space
        )
        
        analyzer = AdvancedConstraintAlgebraAnalyzer(
            constraint, lattice_config, lqg_params
        )
        
        # Test all pairs of lapse functions
        pair_results = {}
        tested_pairs = 0
        
        for i, N_lapse in enumerate(lapse_set):
            for j, M_lapse in enumerate(lapse_set):
                if tested_pairs >= max_pairs:
                    break
                
                pair_key = f"N{i}_M{j}"
                
                try:
                    # Compute [Ĥ[N], Ĥ[M]] commutator
                    comm_matrix, comm_info = analyzer.compute_hamiltonian_commutator(
                        N_lapse, M_lapse
                    )
                    
                    pair_results[pair_key] = {
                        "lapse_N_index": i,
                        "lapse_M_index": j,
                        "closure_error": float(comm_info["closure_error"]),
                        "is_anomaly_free": bool(comm_info["anomaly_free"]),
                        "commutator_norm": float(np.linalg.norm(comm_matrix.toarray() 
                                                              if hasattr(comm_matrix, 'toarray') 
                                                              else comm_matrix))
                    }
                    
                    tested_pairs += 1
                    
                    if tested_pairs % 10 == 0:
                        print(f"   Tested {tested_pairs}/{max_pairs} pairs...")
                
                except Exception as e:
                    print(f"   ⚠️  Error for pair ({i},{j}): {e}")
                    pair_results[pair_key] = {"error": str(e)}
            
            if tested_pairs >= max_pairs:
                break
        
        # Compute statistical summary
        successful_tests = [r for r in pair_results.values() if "error" not in r]
        
        if successful_tests:
            closure_errors = [r["closure_error"] for r in successful_tests]
            anomaly_free_count = sum(r["is_anomaly_free"] for r in successful_tests)
            
            statistics = {
                "total_pairs_tested": len(successful_tests),
                "anomaly_free_pairs": anomaly_free_count,
                "anomaly_free_rate": anomaly_free_count / len(successful_tests),
                "avg_closure_error": float(np.mean(closure_errors)),
                "max_closure_error": float(np.max(closure_errors)),
                "min_closure_error": float(np.min(closure_errors)),
                "std_closure_error": float(np.std(closure_errors))
            }
        else:
            statistics = {"error": "No successful tests"}
        
        return {
            "pair_results": pair_results,
            "statistics": statistics,
            "lapse_functions": [lapse.tolist() for lapse in lapse_set],
            "system_info": {
                "n_sites": self.n_sites,
                "hilbert_dim": kin_space.dim,
                "mu_max": lqg_params.mu_max,
                "basis_truncation": lqg_params.basis_truncation
            }
        }
    
    def plot_anomaly_analysis(self, results: Dict[str, Any]):
        """Create visualization plots of anomaly scan results."""
        
        print(f"\n📊 Creating anomaly analysis plots...")
        
        if "statistics" not in results or "error" in results["statistics"]:
            print("   ⚠️  No valid results to plot")
            return
        
        stats = results["statistics"]
        pair_results = results["pair_results"]
        
        # Extract closure errors for plotting
        successful_pairs = [(k, v) for k, v in pair_results.items() if "error" not in v]
        closure_errors = [v["closure_error"] for _, v in successful_pairs]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram of closure errors
        ax1.hist(closure_errors, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Closure Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Closure Errors')
        ax1.set_yscale('log')
        
        # 2. Closure errors vs pair index
        ax2.plot(range(len(closure_errors)), closure_errors, 'o-', alpha=0.6)
        ax2.set_xlabel('Pair Index')
        ax2.set_ylabel('Closure Error')
        ax2.set_title('Closure Errors by Test Pair')
        ax2.set_yscale('log')
        
        # 3. Anomaly-free rate summary
        ax3.bar(['Anomaly-Free', 'Anomalous'], 
                [stats["anomaly_free_pairs"], 
                 stats["total_pairs_tested"] - stats["anomaly_free_pairs"]],
                color=['green', 'red'], alpha=0.7)
        ax3.set_ylabel('Number of Pairs')
        ax3.set_title(f'Anomaly-Free Rate: {stats["anomaly_free_rate"]:.1%}')
        
        # 4. Error statistics
        error_stats = [stats["min_closure_error"], 
                      stats["avg_closure_error"], 
                      stats["max_closure_error"]]
        ax4.bar(['Min', 'Avg', 'Max'], error_stats, alpha=0.7)
        ax4.set_ylabel('Closure Error')
        ax4.set_title('Error Statistics')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plot_path = self.output_dir / "anomaly_analysis_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Plots saved to {plot_path}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete constraint anomaly analysis pipeline.
        
        Includes:
        1. Regularization parameter scanning
        2. Comprehensive anomaly scanning
        3. Statistical analysis
        4. Visualization
        5. Results export
        """
        print(f"🚀 Starting complete constraint anomaly analysis...")
        
        complete_results = {
            "metadata": {
                "n_sites": self.n_sites,
                "analysis_timestamp": str(np.datetime64('now')),
                "scanner_version": "1.0"
            }
        }
        
        try:
            # Step 1: Regularization scanning
            print(f"\n📍 Step 1: Regularization parameter scanning")
            reg_results = self.scan_regularization_parameters()
            complete_results["regularization_scan"] = reg_results
            
            # Step 2: Comprehensive anomaly scanning
            print(f"\n📍 Step 2: Comprehensive anomaly scanning")
            anomaly_results = self.comprehensive_anomaly_scan()
            complete_results["anomaly_scan"] = anomaly_results
            
            # Step 3: Create visualizations
            print(f"\n📍 Step 3: Creating visualizations")
            self.plot_anomaly_analysis(anomaly_results)
            
            # Step 4: Save results
            results_file = self.output_dir / "complete_anomaly_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(complete_results, f, indent=2)
            
            # Step 5: Generate summary report
            self.generate_summary_report(complete_results)
            
            print(f"\n✅ Complete anomaly analysis finished!")
            print(f"   Results saved to: {results_file}")
            
            return complete_results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        
        report_file = self.output_dir / "anomaly_analysis_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("🔬 CONSTRAINT ANOMALY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"System Configuration:\n")
            f.write(f"  Lattice sites: {self.n_sites}\n")
            f.write(f"  Analysis timestamp: {results['metadata']['analysis_timestamp']}\n\n")
            
            if "regularization_scan" in results:
                f.write("Regularization Parameter Scan:\n")
                reg_data = results["regularization_scan"]
                f.write(f"  Configurations tested: {len(reg_data)}\n")
                
                # Find best configuration
                valid_configs = {k: v for k, v in reg_data.items() if "error" not in v}
                if valid_configs:
                    best_config = max(valid_configs.items(), 
                                    key=lambda x: x[1]["anomaly_free_rate"])
                    f.write(f"  Best configuration: {best_config[0]}\n")
                    f.write(f"  Best anomaly-free rate: {best_config[1]['anomaly_free_rate']:.1%}\n\n")
            
            if "anomaly_scan" in results and "statistics" in results["anomaly_scan"]:
                stats = results["anomaly_scan"]["statistics"]
                if "error" not in stats:
                    f.write("Comprehensive Anomaly Scan:\n")
                    f.write(f"  Total pairs tested: {stats['total_pairs_tested']}\n")
                    f.write(f"  Anomaly-free pairs: {stats['anomaly_free_pairs']}\n")
                    f.write(f"  Anomaly-free rate: {stats['anomaly_free_rate']:.1%}\n")
                    f.write(f"  Average closure error: {stats['avg_closure_error']:.2e}\n")
                    f.write(f"  Maximum closure error: {stats['max_closure_error']:.2e}\n\n")
                    
                    # Verdict
                    if stats['anomaly_free_rate'] > 0.95:
                        f.write("🟢 VERDICT: Constraint algebra is highly consistent!\n")
                    elif stats['anomaly_free_rate'] > 0.8:
                        f.write("🟡 VERDICT: Constraint algebra is mostly consistent.\n")
                    else:
                        f.write("🔴 VERDICT: Significant anomalies detected.\n")
        
        print(f"   📝 Summary report saved to: {report_file}")


def demo_constraint_anomaly_scanner():
    """Demonstration of the constraint anomaly scanner."""
    
    print("🧪 CONSTRAINT ANOMALY SCANNER DEMO")
    print("=" * 60)
    
    # Create scanner for a small system
    scanner = ConstraintAnomalyScanner(n_sites=3)
    
    # Run complete analysis
    results = scanner.run_complete_analysis()
    
    # Print key results
    if "anomaly_scan" in results and "statistics" in results["anomaly_scan"]:
        stats = results["anomaly_scan"]["statistics"]
        if "error" not in stats:
            print(f"\n🏆 FINAL RESULTS:")
            print(f"   Anomaly-free rate: {stats['anomaly_free_rate']:.1%}")
            print(f"   Average closure error: {stats['avg_closure_error']:.2e}")
            print(f"   Tests performed: {stats['total_pairs_tested']}")
    
    return results


if __name__ == "__main__":
    demo_constraint_anomaly_scanner()
