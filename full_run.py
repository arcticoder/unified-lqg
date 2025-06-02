#!/usr/bin/env python3
"""
Full LQG Framework Integration - Five Major Extensions

This script orchestrates the complete LQG framework with all five major extensions:

A. Additional Matter Fields (Maxwell + Dirac) 
B. Advanced Constraint Algebra Verification
C. Automated Lattice Refinement Framework  
D. Angular Perturbation Extension (Beyond Spherical Symmetry)
E. Spin-Foam Cross-Validation

Produces comprehensive analysis results and exports to JSON format.
"""

import json
import numpy as np
import scipy.sparse.linalg as spla
from pathlib import Path
import sys
import argparse
from typing import Dict, Any

# Import all LQG framework components
try:
    from lqg_fixed_components import (
        LatticeConfiguration,
        LQGParameters, 
        KinematicalHilbertSpace,
        MidisuperspaceHamiltonianConstraint
    )
    from lqg_additional_matter import MaxwellField, DiracField, AdditionalMatterFieldsDemo
    from constraint_algebra import AdvancedConstraintAlgebraAnalyzer
    from refinement_framework import run_lqg_for_size, analyze_convergence, generate_convergence_plots
    from angular_perturbation import (
        SphericalHarmonicMode,
        ExtendedKinematicalHilbertSpace, 
        ExtendedMidisuperspaceHamiltonianConstraint
    )
    from spinfoam_validation import SpinFoamCrossValidationDemo
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all LQG framework modules are available.")
    sys.exit(1)


def main():
    """Main orchestrator for the complete LQG framework."""
    
    print("🌌 COMPREHENSIVE LQG FRAMEWORK - FIVE MAJOR EXTENSIONS")
    print("=" * 80)
    print("Executing complete quantum gravity framework for warp drive studies")
    print("Extensions: Matter Fields + Constraint Algebra + Refinement + Angular + Spin-Foam\n")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup output directory
    OUTPUT_DIR = Path("outputs/full_run")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Initialize results storage
    results = {}
    
    try:
        # ===================================================================
        # A. ADDITIONAL MATTER FIELDS (Maxwell + Dirac + Phantom)
        # ===================================================================
        print("\n📡 SECTION A: ADDITIONAL MATTER FIELDS")
        print("=" * 60)
        
        matter_results = run_multi_field_integration(args.n_sites, OUTPUT_DIR)
        results["multi_field_backreaction"] = matter_results
        
        # ===================================================================  
        # B. CONSTRAINT ALGEBRA VERIFICATION
        # ===================================================================
        print("\n🔬 SECTION B: CONSTRAINT ALGEBRA VERIFICATION")
        print("=" * 60)
        
        constraint_results = run_constraint_algebra_analysis(args.n_sites, OUTPUT_DIR)
        results["constraint_algebra"] = constraint_results
        
        # ===================================================================
        # C. AUTOMATED LATTICE REFINEMENT
        # ===================================================================
        print("\n📏 SECTION C: AUTOMATED LATTICE REFINEMENT")  
        print("=" * 60)
        
        if args.max_n_refinement > args.n_sites:
            refinement_results = run_lattice_refinement_study(
                args.n_sites, args.max_n_refinement, OUTPUT_DIR
            )
            results["lattice_refinement"] = refinement_results
        else:
            print("   Skipping refinement (max_n_refinement <= n_sites)")
            results["lattice_refinement"] = {"skipped": True}
        
        # ===================================================================
        # D. ANGULAR PERTURBATION EXTENSION  
        # ===================================================================
        print("\n🌀 SECTION D: ANGULAR PERTURBATION EXTENSION")
        print("=" * 60)
        
        if args.max_l > 0:
            angular_results = run_angular_perturbation_analysis(
                args.n_sites, args.max_l, OUTPUT_DIR
            )
            results["angular_perturbation"] = angular_results
        else:
            print("   Skipping angular perturbations (max_l = 0)")
            results["angular_perturbation"] = {"skipped": True}
        
        # ===================================================================
        # E. SPIN-FOAM CROSS-VALIDATION
        # ===================================================================
        print("\n🕸️  SECTION E: SPIN-FOAM CROSS-VALIDATION")
        print("=" * 60)
        
        spinfoam_results = run_spinfoam_validation(args.n_sites, OUTPUT_DIR)
        results["spinfoam_cross_validation"] = spinfoam_results
        
        # ===================================================================
        # FINAL SUMMARY AND EXPORT
        # ===================================================================
        print("\n📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        # Export comprehensive results
        final_output_file = OUTPUT_DIR / "final_summary.json"
        with open(final_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print_final_summary(results)
        print(f"\n✅ Complete framework execution finished!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"📄 Final summary: {final_output_file}")
        
    except Exception as e:
        print(f"\n❌ Framework execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_multi_field_integration(n_sites: int, output_dir: Path) -> Dict[str, Any]:
    """Run multi-field (Maxwell + Dirac + phantom) integration."""
    
    print("Integrating Maxwell + Dirac + phantom scalar fields...")
    
    # Create multi-field demo
    demo = AdditionalMatterFieldsDemo(n_sites)
    
    # Add Maxwell field with synthetic data
    print("   Setting up Maxwell field...")
    maxwell = demo.add_maxwell_field(
        A_r_data=[0.0, 0.02, 0.005] if n_sites == 3 else [0.01 * np.sin(i) for i in range(n_sites)],
        pi_EM_data=[0.0, 0.004, 0.001] if n_sites == 3 else [0.002 * np.cos(i) for i in range(n_sites)]
    )
    
    # Add Dirac field with synthetic data  
    print("   Setting up Dirac field...")
    if n_sites == 3:
        psi1_data = [0.1+0.05j, 0.05+0.02j, 0.02+0.01j]
        psi2_data = [0.05+0.02j, 0.02+0.01j, 0.01+0.005j]
    else:
        psi1_data = [0.1 * np.exp(1j * i * 0.1) for i in range(n_sites)]
        psi2_data = [0.05 * np.exp(1j * i * 0.15) for i in range(n_sites)]
    
    dirac = demo.add_dirac_field(psi1_data, psi2_data, mass=0.1)
    
    # Create mock Hilbert space
    class MockHilbertSpace:
        def __init__(self, dim):
            self.dim = dim
    
    hilbert_space = MockHilbertSpace(dim=min(100, n_sites * 20))
    
    # Build total stress-energy operator
    print("   Building total stress-energy operator...")
    total_T00 = demo.compute_total_stress_energy(hilbert_space, include_phantom=True)
    
    # Create mock ground state
    ground_state = np.random.random(hilbert_space.dim) + 1j * np.random.random(hilbert_space.dim)
    ground_state = ground_state / np.linalg.norm(ground_state)
    
    # Compute expectation values
    T00_total = (ground_state.conj().T @ total_T00 @ ground_state).real
    
    # Break down by components
    T00_phantom = demo._build_phantom_stress_energy(hilbert_space)
    T00_maxwell = maxwell.compute_stress_energy_operator(hilbert_space)
    T00_dirac = dirac.compute_stress_energy_operator(hilbert_space)
    
    T00_phantom_exp = (ground_state.conj().T @ T00_phantom @ ground_state).real
    T00_maxwell_exp = (ground_state.conj().T @ T00_maxwell @ ground_state).real  
    T00_dirac_exp = (ground_state.conj().T @ T00_dirac @ ground_state).real
    
    results = {
        "T00_phantom": float(T00_phantom_exp),
        "T00_maxwell": float(T00_maxwell_exp),
        "T00_dirac": float(T00_dirac_exp),
        "T00_total": float(T00_total),
        "hilbert_dimension": hilbert_space.dim,
        "operator_nnz": total_T00.nnz,
        "n_sites": n_sites
    }
    
    # Export detailed results
    demo.export_stress_energy_data(hilbert_space, ground_state, 
                                  str(output_dir / "multi_field_detailed.json"))
    
    with open(output_dir / "multi_field_backreaction.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ⟨T⁰⁰_phantom⟩ = {T00_phantom_exp:.6f}")
    print(f"   ⟨T⁰⁰_Maxwell⟩ = {T00_maxwell_exp:.6f}")
    print(f"   ⟨T⁰⁰_Dirac⟩ = {T00_dirac_exp:.6f}")
    print(f"   ⟨T⁰⁰_total⟩ = {T00_total:.6f}")
    
    return results


def run_constraint_algebra_analysis(n_sites: int, output_dir: Path) -> Dict[str, Any]:
    """Run constraint algebra verification."""
    
    print("Analyzing LQG constraint algebra closure...")
    
    # Setup basic LQG components
    lattice_config = LatticeConfiguration(n_sites=n_sites, throat_radius=1.0)
    lqg_params = LQGParameters(mu_max=1, nu_max=1, basis_truncation=min(100, n_sites * 20))
      # Build kinematical Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
      # Build Hamiltonian constraint (with matter fields)
    constraint = MidisuperspaceHamiltonianConstraint(lattice_config, lqg_params, kin_space)
    
    # Synthetic classical data
    E_x = np.ones(n_sites)
    E_phi = np.ones(n_sites) * 1.1
    K_x = np.zeros(n_sites)
    K_phi = np.zeros(n_sites)
    scalar_field = np.ones(n_sites) * 0.1
    scalar_momentum = np.zeros(n_sites)
    
    # Build constraint matrix
    H_matrix = constraint.construct_full_hamiltonian(
        E_x, E_phi, K_x, K_phi, scalar_field, scalar_momentum
    )
    
    # Set up constraint algebra analyzer
    analyzer = AdvancedConstraintAlgebraAnalyzer(constraint, lattice_config, lqg_params)
    
    # Run comprehensive closure verification
    closure_results = analyzer.verify_constraint_closure(test_multiple_lapse_pairs=True)
    
    # Export results
    with open(output_dir / "constraint_algebra.json", 'w') as f:
        json.dump(closure_results, f, indent=2)
    
    print(f"   Constraint algebra tests: {closure_results['total_tests']}")
    print(f"   Anomaly-free rate: {closure_results['anomaly_free_rate']:.1%}")
    print(f"   Average closure error: {closure_results['avg_closure_error']:.2e}")
    
    return closure_results


def run_lattice_refinement_study(n_start: int, n_max: int, output_dir: Path) -> Dict[str, Any]:
    """Run lattice refinement study for multiple N values."""
    
    print(f"Running lattice refinement study: N = {n_start} to {n_max}...")
    
    # Generate N values to test
    N_values = list(range(n_start, min(n_max + 1, n_start + 6), 2))  # Limit to avoid memory issues
    print(f"   Testing N values: {N_values}")
    
    # Base configuration
    base_config = {
        'throat_radius': 1.0,
        'E_x_classical': None,  # Will be generated per size
        'E_phi_classical': None,
        'scalar_field': None,
        'scalar_momentum': None
    }
    
    lqg_params = LQGParameters(mu_max=1, nu_max=1, basis_truncation=200)
    
    # Run refinement study
    refinement_results = {}
    for N in N_values:
        try:
            # Generate synthetic data for this N
            config_N = base_config.copy()
            config_N['E_x_classical'] = np.ones(N)
            config_N['E_phi_classical'] = np.ones(N) * 1.1
            config_N['scalar_field'] = np.ones(N) * 0.1
            config_N['scalar_momentum'] = np.zeros(N)
            
            # Run LQG computation
            obs = run_lqg_for_size(N, config_N, lqg_params)
            refinement_results[N] = obs
            
            print(f"   N={N}: ω²_min = {obs['omega_min_squared']:.3e}, "
                  f"⟨T⁰⁰⟩ = {obs.get('total_stress_energy', 0):.3e}")
            
        except Exception as e:
            print(f"   ⚠️  Failed for N={N}: {e}")
            refinement_results[N] = {"error": str(e)}
    
    # Analyze convergence
    convergence_analysis = analyze_convergence(refinement_results)
    
    # Generate plots if possible
    try:
        generate_convergence_plots(refinement_results, convergence_analysis, 
                                 output_dir / "refinement_convergence.png")
    except Exception as e:
        print(f"   ⚠️  Plot generation failed: {e}")
    
    # Export results
    results = {
        "N_values": N_values,
        "refinement_data": refinement_results,
        "convergence_analysis": convergence_analysis
    }
    
    with open(output_dir / "lattice_refinement.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Convergence analysis completed for {len(N_values)} lattice sizes")
    
    return results


def run_angular_perturbation_analysis(n_sites: int, max_l: int, output_dir: Path) -> Dict[str, Any]:
    """Run angular perturbation analysis."""
    
    print(f"Analyzing angular perturbations up to l_max = {max_l}...")
    
    # Setup angular modes
    angular_modes = []
    for l in range(1, max_l + 1):
        for m in range(-l, l + 1):
            if len(angular_modes) < 3:  # Limit number of modes
                mode = SphericalHarmonicMode(l=l, m=m, amplitude=0.1, alpha_max=1)
                angular_modes.append(mode)
    
    print(f"   Testing {len(angular_modes)} angular modes")
    
    # Setup base configuration
    lattice_config = LatticeConfiguration(n_sites=n_sites, throat_radius=1.0)
    lqg_params = LQGParameters(mu_max=1, nu_max=1, basis_truncation=50)  # Smaller for angular extension
    
    try:
        # Build extended Hilbert space
        ext_space = ExtendedKinematicalHilbertSpace(lattice_config, lqg_params, angular_modes)
        print(f"   Extended Hilbert space dimension: {ext_space.dim}")
        
        # Build extended Hamiltonian
        ext_constraint = ExtendedMidisuperspaceHamiltonianConstraint(
            ext_space, lattice_config, lqg_params
        )
        H_ext = ext_constraint.build_extended_hamiltonian()
        
        # Compute spectrum
        k = min(3, H_ext.shape[0] - 1)
        if k > 0:
            eigenvals, _ = spla.eigs(H_ext, k=k, which='SR')
            eigenvals = np.real(eigenvals)
            eigenvals.sort()
        else:
            eigenvals = [0.0]
          # Compare to purely radial case
        radial_space = KinematicalHilbertSpace(lattice_config, lqg_params)
        radial_constraint = MidisuperspaceHamiltonianConstraint(lattice_config, lqg_params, radial_space)
        
        # Synthetic data for radial case
        E_x = np.ones(n_sites)
        E_phi = np.ones(n_sites) * 1.1
        H_radial = radial_constraint.construct_full_hamiltonian(
            E_x, E_phi, np.zeros(n_sites), np.zeros(n_sites), 
            np.ones(n_sites) * 0.1, np.zeros(n_sites)
        )
        
        k_radial = min(3, H_radial.shape[0] - 1)
        if k_radial > 0:
            eigenvals_radial, _ = spla.eigs(H_radial, k=k_radial, which='SR')
            eigenvals_radial = np.real(eigenvals_radial)
            eigenvals_radial.sort()
        else:
            eigenvals_radial = [0.0]
        
        # Angular shift analysis
        angular_shift = eigenvals[0] - eigenvals_radial[0] if len(eigenvals) > 0 and len(eigenvals_radial) > 0 else 0.0
        
        results = {
            "energy_scale": float(eigenvals[0]) if len(eigenvals) > 0 else 0.0,
            "radial_energy_scale": float(eigenvals_radial[0]) if len(eigenvals_radial) > 0 else 0.0,
            "angular_shift": float(angular_shift),
            "extended_dimension": ext_space.dim,
            "radial_dimension": radial_space.dim,
            "frequency_gaps": [float(gap) for gap in np.diff(eigenvals[:3])] if len(eigenvals) > 2 else [],
            "angular_modes_tested": len(angular_modes),
            "max_l": max_l
        }
        
    except Exception as e:
        print(f"   ⚠️  Angular perturbation analysis failed: {e}")
        results = {
            "error": str(e),
            "max_l": max_l,
            "angular_modes_tested": len(angular_modes)
        }
    
    # Export results
    with open(output_dir / "angular_perturbation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    if "error" not in results:
        print(f"   Energy scale (radial+angular): {results['energy_scale']:.3e}")
        print(f"   Angular shift: {results['angular_shift']:.3e}")
        print(f"   Extended dimension: {results['extended_dimension']}")
    
    return results


def run_spinfoam_validation(n_sites: int, output_dir: Path) -> Dict[str, Any]:
    """Run spin-foam cross-validation."""
    
    print("Running spin-foam cross-validation...")
    
    try:
        # Create spin-foam validation demo
        spinfoam_demo = SpinFoamCrossValidationDemo(n_sites=n_sites)
        
        # Setup canonical reference
        canonical_obs = spinfoam_demo.setup_canonical_reference()
        
        # Build simplified EPRL amplitude
        sf_data = spinfoam_demo.build_simplified_eprl_amplitude()
        
        # Map spin-foam to canonical
        mapping = spinfoam_demo.map_spinfoam_to_canonical()
        
        # Compare observables
        comparison = spinfoam_demo.compare_observables()
        
        results = {
            "relative_error": comparison.get("relative_error", 0.0),
            "is_consistent": comparison.get("is_consistent", False),
            "tolerance": comparison.get("tolerance", 0.1),
            "canonical_energy": canonical_obs.get("ground_state_energy", 0.0),
            "spinfoam_configs": len(sf_data.get("spin_assignments", [])),
            "n_sites": n_sites
        }
        
        print(f"   Spin-foam configurations tested: {results['spinfoam_configs']}")
        print(f"   Relative error: {results['relative_error']:.1%}")
        print(f"   Consistency check: {'✓' if results['is_consistent'] else '✗'}")
        
    except Exception as e:
        print(f"   ⚠️  Spin-foam validation failed: {e}")
        results = {
            "error": str(e),
            "n_sites": n_sites
        }
    
    # Export results
    with open(output_dir / "spinfoam_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def print_final_summary(results: Dict[str, Any]):
    """Print comprehensive final summary."""
    
    print("🏆 COMPREHENSIVE LQG FRAMEWORK EXECUTION SUMMARY")
    print("-" * 60)
    
    # Multi-field backreaction
    if "multi_field_backreaction" in results:
        mf = results["multi_field_backreaction"]
        print(f"📡 Multi-Field Integration:")
        print(f"   ⟨T⁰⁰_phantom⟩ = {mf.get('T00_phantom', 0):.6f}")
        print(f"   ⟨T⁰⁰_Maxwell⟩ = {mf.get('T00_maxwell', 0):.6f}")
        print(f"   ⟨T⁰⁰_Dirac⟩ = {mf.get('T00_dirac', 0):.6f}")
        print(f"   ⟨T⁰⁰_total⟩ = {mf.get('T00_total', 0):.6f}")
    
    # Constraint algebra
    if "constraint_algebra" in results:
        ca = results["constraint_algebra"] 
        print(f"\n🔬 Constraint Algebra:")
        print(f"   Anomaly-free rate: {ca.get('anomaly_free_rate', 0):.1%}")
        print(f"   Average closure error: {ca.get('avg_closure_error', 0):.2e}")
        print(f"   Status: {'✅ PASSED' if ca.get('overall_anomaly_free', False) else '❌ FAILED'}")
    
    # Lattice refinement
    if "lattice_refinement" in results and not results["lattice_refinement"].get("skipped"):
        lr = results["lattice_refinement"]
        conv = lr.get("convergence_analysis", {})
        print(f"\n📏 Lattice Refinement:")
        print(f"   N values tested: {lr.get('N_values', [])}")
        print(f"   Continuum limit: {conv.get('omega_continuum_estimate', 'N/A')}")
    
    # Angular perturbations
    if "angular_perturbation" in results and not results["angular_perturbation"].get("skipped"):
        ap = results["angular_perturbation"]
        if "error" not in ap:
            print(f"\n🌀 Angular Perturbations:")
            print(f"   Energy scale: {ap.get('energy_scale', 0):.3e}")
            print(f"   Angular shift: {ap.get('angular_shift', 0):.3e}")
            print(f"   Extended dimension: {ap.get('extended_dimension', 0)}")
    
    # Spin-foam validation
    if "spinfoam_cross_validation" in results:
        sf = results["spinfoam_cross_validation"]
        if "error" not in sf:
            print(f"\n🕸️  Spin-Foam Validation:")
            print(f"   Relative error: {sf.get('relative_error', 0):.1%}")
            print(f"   Consistency: {'✓' if sf.get('is_consistent', False) else '✗'}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Complete LQG Framework Integration")
    
    parser.add_argument("--n_sites", type=int, default=3,
                       help="Number of lattice sites (default: 3)")
    parser.add_argument("--max_l", type=int, default=1, 
                       help="Maximum angular momentum for perturbations (default: 1)")
    parser.add_argument("--max_n_refinement", type=int, default=7,
                       help="Maximum N for lattice refinement study (default: 7)")
    parser.add_argument("--include_dirac", action="store_true", default=True,
                       help="Include Dirac field (default: True)")
    
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
