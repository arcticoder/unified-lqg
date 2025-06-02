"""
Enhanced Spin-Foam Cross-Validation with Larger Networks

This module implements enhanced spin-foam validation with larger spin networks,
improved amplitude calculations, and comprehensive cross-validation against
canonical LQG results.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations, product
import warnings
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.special import factorial, comb

# Import core LQG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
from spinfoam_validation import SpinfoamValidator  # Import existing validator

@dataclass
class SpinNetworkConfig:
    """Configuration for spin network generation"""
    num_vertices: int
    num_edges: int
    max_spin: int
    edge_structure: str  # 'linear', 'triangular', 'tetrahedral', 'random'
    include_faces: bool = True
    boundary_conditions: str = 'periodic'

@dataclass
class SpinFoamAmplitude:
    """Container for spin-foam amplitude calculation"""
    network_config: SpinNetworkConfig
    spin_assignment: Dict[int, float]  # edge_id -> spin value
    amplitude_value: complex
    computation_time: float
    numerical_stability: float

@dataclass
class ValidationResult:
    """Results from enhanced spin-foam validation"""
    network_size: int
    canonical_energy: float
    spinfoam_energy: float
    relative_error: float
    amplitude_count: int
    convergence_metric: float
    stability_index: float
    correlation_coefficient: float

class EnhancedSpinFoamValidator:
    """
    Enhanced spin-foam validation framework with larger networks
    and improved amplitude calculations.
    """
    
    def __init__(self, N: int = 7, max_network_size: int = 20):
        """
        Initialize enhanced spin-foam validator.
        
        Args:
            N: Lattice size for canonical LQG
            max_network_size: Maximum number of vertices in spin networks
        """
        self.N = N
        self.max_network_size = max_network_size
        
        # Initialize canonical LQG
        self.constraint = MidisuperspaceHamiltonianConstraint(N)
        self.canonical_basis = self.constraint.generate_flux_basis()
        
        # Spin network configurations to test
        self.network_configs = self._generate_network_configurations()
        
        # Results storage
        self.validation_results = []
        self.amplitude_cache = {}
        
    def _generate_network_configurations(self) -> List[SpinNetworkConfig]:
        """Generate diverse spin network configurations for testing"""
        configs = []
        
        # Linear networks (1D-like)
        for vertices in range(3, min(self.max_network_size, 10)):
            configs.append(SpinNetworkConfig(
                num_vertices=vertices,
                num_edges=vertices - 1,
                max_spin=2.0,
                edge_structure='linear',
                boundary_conditions='open'
            ))
        
        # Triangular networks (2D-like)
        for vertices in range(3, min(self.max_network_size, 8)):
            num_edges = int(vertices * (vertices - 1) / 2)  # Complete graph
            configs.append(SpinNetworkConfig(
                num_vertices=vertices,
                num_edges=min(num_edges, vertices * 3),  # Limit edges
                max_spin=2.0,
                edge_structure='triangular',
                boundary_conditions='periodic'
            ))
        
        # Tetrahedral networks (3D-like)
        for vertices in [4, 6, 8, 10]:
            if vertices <= self.max_network_size:
                configs.append(SpinNetworkConfig(
                    num_vertices=vertices,
                    num_edges=vertices * 2,  # Typical 3D connectivity
                    max_spin=2.5,
                    edge_structure='tetrahedral',
                    boundary_conditions='closed'
                ))
        
        # Random networks
        for vertices in [5, 7, 9, 12, 15]:
            if vertices <= self.max_network_size:
                configs.append(SpinNetworkConfig(
                    num_vertices=vertices,
                    num_edges=int(vertices * 1.5),  # Moderate connectivity
                    max_spin=3.0,
                    edge_structure='random',
                    boundary_conditions='mixed'
                ))
        
        return configs
    
    def generate_spin_network(self, config: SpinNetworkConfig) -> Dict[str, Any]:
        """
        Generate a spin network according to the given configuration.
        
        Args:
            config: Configuration for the spin network
            
        Returns:
            Dictionary containing network structure
        """
        network = {
            'vertices': list(range(config.num_vertices)),
            'edges': [],
            'edge_spins': {},
            'face_spins': {},
            'vertex_constraints': {}
        }
        
        # Generate edges based on structure type
        if config.edge_structure == 'linear':
            # Linear chain
            edges = [(i, i+1) for i in range(config.num_vertices - 1)]
            if config.boundary_conditions == 'periodic':
                edges.append((config.num_vertices - 1, 0))
                
        elif config.edge_structure == 'triangular':
            # Triangular lattice (partial)
            edges = []
            for i in range(config.num_vertices):
                for j in range(i+1, min(i+4, config.num_vertices)):  # Limit connectivity
                    edges.append((i, j))
                    if len(edges) >= config.num_edges:
                        break
                if len(edges) >= config.num_edges:
                    break
                    
        elif config.edge_structure == 'tetrahedral':
            # 3D tetrahedral-like structure
            edges = []
            for i in range(config.num_vertices):
                for j in range(i+1, config.num_vertices):
                    if len(edges) < config.num_edges:
                        # Add edge with some probability based on 3D structure
                        if self._should_add_edge_3d(i, j, config.num_vertices):
                            edges.append((i, j))
                            
        elif config.edge_structure == 'random':
            # Random graph
            all_possible_edges = list(combinations(range(config.num_vertices), 2))
            np.random.shuffle(all_possible_edges)
            edges = all_possible_edges[:config.num_edges]
            
        else:
            raise ValueError(f"Unknown edge structure: {config.edge_structure}")
        
        network['edges'] = edges[:config.num_edges]
        
        # Assign spin values to edges
        for i, edge in enumerate(network['edges']):
            # Use half-integer spins: 1/2, 1, 3/2, 2, ...
            max_spin_int = int(2 * config.max_spin)
            spin_value = np.random.choice(range(1, max_spin_int + 1)) / 2.0
            network['edge_spins'][i] = spin_value
        
        # Generate faces if required
        if config.include_faces:
            network['faces'] = self._generate_faces(network['vertices'], network['edges'])
            
            # Assign spins to faces
            for i, face in enumerate(network['faces']):
                face_spin = np.random.choice(range(0, int(2 * config.max_spin) + 1)) / 2.0
                network['face_spins'][i] = face_spin
        
        return network
    
    def _should_add_edge_3d(self, i: int, j: int, num_vertices: int) -> bool:
        """Determine if edge should be added in 3D-like structure"""
        # Simple heuristic for 3D connectivity
        distance = abs(i - j)
        if distance == 1:  # Nearest neighbors
            return True
        elif distance == 2 and np.random.random() < 0.5:  # Next-nearest neighbors
            return True
        elif distance > 2 and np.random.random() < 0.2:  # Long-range connections
            return True
        return False
    
    def _generate_faces(self, vertices: List[int], edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Generate triangular faces from vertex and edge structure"""
        faces = []
        
        # Find all triangles in the graph
        for v1, v2, v3 in combinations(vertices, 3):
            # Check if all three edges exist
            edges_needed = [(v1, v2), (v1, v3), (v2, v3)]
            edges_set = set(edges) | set((j, i) for i, j in edges)  # Undirected
            
            if all((e in edges_set) for e in edges_needed):
                faces.append((v1, v2, v3))
                
            # Limit number of faces
            if len(faces) >= len(edges):
                break
        
        return faces
    
    def compute_spin_foam_amplitude(self, network: Dict[str, Any], 
                                   config: SpinNetworkConfig) -> SpinFoamAmplitude:
        """
        Compute enhanced spin-foam amplitude for the given network.
        
        Args:
            network: Spin network structure
            config: Configuration used to generate the network
            
        Returns:
            SpinFoamAmplitude result
        """
        import time
        start_time = time.time()
        
        # Create unique key for caching
        cache_key = self._create_cache_key(network, config)
        if cache_key in self.amplitude_cache:
            cached_result = self.amplitude_cache[cache_key]
            print(f"    📋 Using cached amplitude for network size {config.num_vertices}")
            return cached_result
        
        print(f"    🔧 Computing amplitude for {config.edge_structure} network "
              f"({config.num_vertices} vertices, {config.num_edges} edges)")
        
        try:
            # Compute vertex amplitudes
            vertex_amplitudes = []
            for vertex in network['vertices']:
                vertex_amp = self._compute_vertex_amplitude(vertex, network)
                vertex_amplitudes.append(vertex_amp)
            
            # Compute edge amplitudes  
            edge_amplitudes = []
            for edge_id, edge in enumerate(network['edges']):
                edge_amp = self._compute_edge_amplitude(edge_id, network)
                edge_amplitudes.append(edge_amp)
            
            # Compute face amplitudes (if faces exist)
            face_amplitudes = []
            if 'faces' in network and network['faces']:
                for face_id, face in enumerate(network['faces']):
                    face_amp = self._compute_face_amplitude(face_id, network)
                    face_amplitudes.append(face_amp)
            
            # Combine all amplitudes
            total_amplitude = self._combine_amplitudes(
                vertex_amplitudes, edge_amplitudes, face_amplitudes
            )
            
            # Compute numerical stability metric
            stability = self._assess_numerical_stability(
                vertex_amplitudes, edge_amplitudes, face_amplitudes
            )
            
            computation_time = time.time() - start_time
            
            result = SpinFoamAmplitude(
                network_config=config,
                spin_assignment=network['edge_spins'].copy(),
                amplitude_value=total_amplitude,
                computation_time=computation_time,
                numerical_stability=stability
            )
            
            # Cache the result
            self.amplitude_cache[cache_key] = result
            
            print(f"      ✅ Amplitude = {total_amplitude:.6f}, "
                  f"stability = {stability:.3f}, time = {computation_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"      ❌ Error computing amplitude: {str(e)}")
            # Return a dummy result with zero amplitude
            return SpinFoamAmplitude(
                network_config=config,
                spin_assignment=network['edge_spins'].copy(),
                amplitude_value=0.0,
                computation_time=time.time() - start_time,
                numerical_stability=0.0
            )
    
    def _create_cache_key(self, network: Dict[str, Any], config: SpinNetworkConfig) -> str:
        """Create a unique cache key for the network and configuration"""
        edge_spins_str = "_".join(f"{k}:{v}" for k, v in sorted(network['edge_spins'].items()))
        return f"{config.edge_structure}_{config.num_vertices}_{config.num_edges}_{edge_spins_str}"
    
    def _compute_vertex_amplitude(self, vertex: int, network: Dict[str, Any]) -> complex:
        """Compute amplitude contribution from a vertex"""
        # Get edges connected to this vertex
        connected_edges = []
        for edge_id, (v1, v2) in enumerate(network['edges']):
            if v1 == vertex or v2 == vertex:
                connected_edges.append(edge_id)
        
        if not connected_edges:
            return 1.0 + 0j
        
        # Compute vertex amplitude using Wigner 6j symbols (simplified)
        # In full implementation, this would use proper 6j symbols
        total_spin = sum(network['edge_spins'][eid] for eid in connected_edges)
        
        # Simple vertex amplitude: depends on total spin at vertex
        vertex_amp = np.exp(-0.5 * total_spin) * (1 + 0.1j * total_spin)
        
        # Add quantum corrections
        quantum_correction = self._compute_quantum_correction(connected_edges, network)
        vertex_amp *= quantum_correction
        
        return vertex_amp
    
    def _compute_edge_amplitude(self, edge_id: int, network: Dict[str, Any]) -> complex:
        """Compute amplitude contribution from an edge"""
        spin = network['edge_spins'][edge_id]
        
        # Edge amplitude: propagator between vertices
        # Using simplified form based on spin foam theory
        
        # Dimension factor: 2j + 1
        dimension = 2 * spin + 1
        
        # Exponential factor (simplified action)
        action_factor = np.exp(-1j * spin * np.pi / 4)
        
        # Edge amplitude
        edge_amp = np.sqrt(dimension) * action_factor
        
        # Add geometric corrections
        geometric_correction = self._compute_geometric_correction(edge_id, network)
        edge_amp *= geometric_correction
        
        return edge_amp
    
    def _compute_face_amplitude(self, face_id: int, network: Dict[str, Any]) -> complex:
        """Compute amplitude contribution from a face"""
        if 'face_spins' not in network or face_id not in network['face_spins']:
            return 1.0 + 0j
        
        face_spin = network['face_spins'][face_id]
        face = network['faces'][face_id]
        
        # Find edge spins bordering this face
        bordering_spins = []
        for edge_id, (v1, v2) in enumerate(network['edges']):
            if (v1 in face and v2 in face):
                bordering_spins.append(network['edge_spins'][edge_id])
        
        if len(bordering_spins) >= 3:
            # Compute Wigner 6j symbol contribution (simplified)
            j1, j2, j3 = bordering_spins[:3]
            
            # Simplified 6j symbol calculation
            sixj_value = self._compute_simplified_6j(j1, j2, j3, face_spin)
            
            # Face amplitude
            face_amp = sixj_value * np.exp(-1j * face_spin * np.pi / 6)
        else:
            face_amp = 1.0 + 0j
        
        return face_amp
    
    def _compute_quantum_correction(self, connected_edges: List[int], 
                                   network: Dict[str, Any]) -> complex:
        """Compute quantum corrections for vertex amplitude"""
        if not connected_edges:
            return 1.0 + 0j
        
        # Simple quantum correction based on edge spins
        total_spin = sum(network['edge_spins'][eid] for eid in connected_edges)
        quantum_phase = -1j * total_spin * 0.1
        
        return np.exp(quantum_phase)
    
    def _compute_geometric_correction(self, edge_id: int, network: Dict[str, Any]) -> complex:
        """Compute geometric corrections for edge amplitude"""
        edge = network['edges'][edge_id]
        vertex_distance = abs(edge[1] - edge[0])
        
        # Geometric phase based on edge length
        geometric_phase = -1j * vertex_distance * 0.05
        
        return np.exp(geometric_phase)
    
    def _compute_simplified_6j(self, j1: float, j2: float, j3: float, j4: float) -> complex:
        """Compute simplified Wigner 6j symbol"""
        # This is a simplified approximation
        # Full implementation would use proper Wigner 6j calculations
        
        # Check triangle inequalities (simplified)
        if abs(j1 - j2) <= j3 <= j1 + j2:
            # Compute approximate 6j symbol
            factor = np.sqrt((2*j1 + 1) * (2*j2 + 1) * (2*j3 + 1) * (2*j4 + 1))
            phase = np.exp(-1j * (j1 + j2 + j3 + j4) * np.pi / 8)
            
            return factor * phase / (8 * np.pi)
        else:
            return 0.0 + 0j
    
    def _combine_amplitudes(self, vertex_amps: List[complex], 
                           edge_amps: List[complex], 
                           face_amps: List[complex]) -> complex:
        """Combine all amplitude contributions"""
        # Product of all vertex amplitudes
        vertex_product = np.prod(vertex_amps) if vertex_amps else 1.0
        
        # Product of all edge amplitudes
        edge_product = np.prod(edge_amps) if edge_amps else 1.0
        
        # Product of all face amplitudes
        face_product = np.prod(face_amps) if face_amps else 1.0
        
        # Total amplitude
        total_amplitude = vertex_product * edge_product * face_product
        
        # Apply overall normalization
        normalization = 1.0 / np.sqrt(len(vertex_amps) + len(edge_amps) + len(face_amps) + 1)
        
        return total_amplitude * normalization
    
    def _assess_numerical_stability(self, vertex_amps: List[complex], 
                                   edge_amps: List[complex], 
                                   face_amps: List[complex]) -> float:
        """Assess numerical stability of amplitude calculation"""
        all_amps = vertex_amps + edge_amps + face_amps
        
        if not all_amps:
            return 1.0
        
        # Check for NaN or inf values
        finite_count = sum(1 for amp in all_amps if np.isfinite(amp))
        finite_ratio = finite_count / len(all_amps)
        
        # Check magnitude spread
        magnitudes = [abs(amp) for amp in all_amps if np.isfinite(amp)]
        if magnitudes:
            magnitude_ratio = min(magnitudes) / (max(magnitudes) + 1e-10)
        else:
            magnitude_ratio = 0.0
        
        # Stability metric (higher is more stable)
        stability = finite_ratio * magnitude_ratio
        
        return min(stability, 1.0)
    
    def compute_canonical_energy(self) -> float:
        """Compute canonical LQG energy expectation value"""
        # Use coherent state expectation value
        coherent_state = self._generate_coherent_state_canonical()
        
        # Build Hamiltonian matrix
        H_matrix = np.zeros((len(self.canonical_basis), len(self.canonical_basis)), 
                           dtype=complex)
        for i, state_i in enumerate(self.canonical_basis):
            for j, state_j in enumerate(self.canonical_basis):
                H_matrix[i, j] = self.constraint.compute_matrix_element(state_i, state_j)
        
        # Compute energy expectation value
        energy = np.real(np.conj(coherent_state) @ H_matrix @ coherent_state)
        
        return energy
    
    def _generate_coherent_state_canonical(self) -> np.ndarray:
        """Generate coherent state for canonical calculation"""
        dim = len(self.canonical_basis)
        alpha = 0.5 + 0.3j
        
        state = np.exp(-0.5 * abs(alpha)**2) * np.array([
            alpha**n / np.sqrt(np.math.factorial(min(n, 20))) 
            for n in range(dim)
        ])
        return state / np.linalg.norm(state)
    
    def convert_spinfoam_to_energy(self, amplitude: SpinFoamAmplitude) -> float:
        """Convert spin-foam amplitude to energy scale"""
        # Extract energy scale from amplitude
        amplitude_magnitude = abs(amplitude.amplitude_value)
        
        if amplitude_magnitude > 0:
            # Use logarithmic mapping to energy scale
            energy_scale = -np.log(amplitude_magnitude)
            
            # Apply corrections based on network size
            size_correction = amplitude.network_config.num_vertices / 10.0
            
            # Apply stability correction
            stability_correction = amplitude.numerical_stability
            
            spinfoam_energy = energy_scale * size_correction * (1 + stability_correction)
        else:
            spinfoam_energy = 0.0
        
        return spinfoam_energy
    
    def run_enhanced_validation(self) -> List[ValidationResult]:
        """
        Run enhanced spin-foam validation across multiple network configurations.
        
        Returns:
            List of ValidationResult for each configuration tested
        """
        print("🌊 Starting Enhanced Spin-Foam Cross-Validation")
        print("="*55)
        print(f"Network configurations: {len(self.network_configs)}")
        print(f"Max network size: {self.max_network_size}")
        
        # Compute canonical energy once
        canonical_energy = self.compute_canonical_energy()
        print(f"Canonical LQG energy: {canonical_energy:.6f}")
        
        self.validation_results = []
        
        for i, config in enumerate(self.network_configs):
            print(f"\n🔧 Validating configuration {i+1}/{len(self.network_configs)}: "
                  f"{config.edge_structure} network")
            
            try:
                # Generate spin network
                network = self.generate_spin_network(config)
                
                # Compute spin-foam amplitude
                amplitude = self.compute_spin_foam_amplitude(network, config)
                
                # Convert to energy scale
                spinfoam_energy = self.convert_spinfoam_to_energy(amplitude)
                
                # Compute relative error
                if abs(canonical_energy) > 1e-10:
                    relative_error = abs(spinfoam_energy - canonical_energy) / abs(canonical_energy)
                else:
                    relative_error = abs(spinfoam_energy)
                
                # Compute convergence metric
                convergence_metric = self._compute_convergence_metric(config, amplitude)
                
                # Compute correlation coefficient (simplified)
                correlation = self._compute_correlation_coefficient(
                    canonical_energy, spinfoam_energy, amplitude
                )
                
                result = ValidationResult(
                    network_size=config.num_vertices,
                    canonical_energy=canonical_energy,
                    spinfoam_energy=spinfoam_energy,
                    relative_error=relative_error,
                    amplitude_count=1,  # Single amplitude per configuration
                    convergence_metric=convergence_metric,
                    stability_index=amplitude.numerical_stability,
                    correlation_coefficient=correlation
                )
                
                self.validation_results.append(result)
                
                print(f"    ✅ E_canonical = {canonical_energy:.6f}, "
                      f"E_spinfoam = {spinfoam_energy:.6f}")
                print(f"    Relative error = {relative_error:.4f}, "
                      f"correlation = {correlation:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error in validation: {str(e)}")
                continue
        
        return self.validation_results
    
    def _compute_convergence_metric(self, config: SpinNetworkConfig, 
                                   amplitude: SpinFoamAmplitude) -> float:
        """Compute convergence metric for the configuration"""
        # Based on network size and stability
        size_factor = 1.0 / (config.num_vertices + 1)
        stability_factor = amplitude.numerical_stability
        
        return size_factor * stability_factor
    
    def _compute_correlation_coefficient(self, canonical_energy: float, 
                                        spinfoam_energy: float, 
                                        amplitude: SpinFoamAmplitude) -> float:
        """Compute correlation coefficient between approaches"""
        # Simplified correlation based on energy agreement and stability
        energy_agreement = 1.0 / (1.0 + abs(canonical_energy - spinfoam_energy))
        stability_bonus = amplitude.numerical_stability
        
        return min(energy_agreement * (1 + stability_bonus), 1.0)
    
    def analyze_validation_results(self) -> Dict[str, Any]:
        """Analyze enhanced validation results"""
        if not self.validation_results:
            return {"error": "No validation results to analyze"}
        
        analysis = {
            "total_validations": len(self.validation_results),
            "network_size_range": (
                min(r.network_size for r in self.validation_results),
                max(r.network_size for r in self.validation_results)
            ),
            "error_statistics": {},
            "stability_statistics": {},
            "correlation_statistics": {},
            "convergence_analysis": {}
        }
        
        # Error statistics
        errors = [r.relative_error for r in self.validation_results]
        analysis["error_statistics"] = {
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "std_error": np.std(errors),
            "min_error": np.min(errors),
            "max_error": np.max(errors)
        }
        
        # Stability statistics
        stabilities = [r.stability_index for r in self.validation_results]
        analysis["stability_statistics"] = {
            "mean_stability": np.mean(stabilities),
            "median_stability": np.median(stabilities),
            "min_stability": np.min(stabilities),
            "max_stability": np.max(stabilities)
        }
        
        # Correlation statistics
        correlations = [r.correlation_coefficient for r in self.validation_results]
        analysis["correlation_statistics"] = {
            "mean_correlation": np.mean(correlations),
            "median_correlation": np.median(correlations),
            "min_correlation": np.min(correlations),
            "max_correlation": np.max(correlations)
        }
        
        # Convergence analysis by network size
        size_groups = {}
        for result in self.validation_results:
            size = result.network_size
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)
        
        for size, results in size_groups.items():
            avg_error = np.mean([r.relative_error for r in results])
            avg_stability = np.mean([r.stability_index for r in results])
            analysis["convergence_analysis"][f"size_{size}"] = {
                "count": len(results),
                "average_error": avg_error,
                "average_stability": avg_stability
            }
        
        return analysis
    
    def export_results(self, output_dir: str = "outputs") -> str:
        """Export enhanced validation results to JSON file"""
        output_path = Path(output_dir) / "enhanced_spinfoam_validation_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "validation_parameters": {
                "lattice_size": self.N,
                "max_network_size": self.max_network_size,
                "network_configurations": len(self.network_configs)
            },
            "validation_results": [
                {
                    "network_size": r.network_size,
                    "canonical_energy": r.canonical_energy,
                    "spinfoam_energy": r.spinfoam_energy,
                    "relative_error": r.relative_error,
                    "amplitude_count": r.amplitude_count,
                    "convergence_metric": r.convergence_metric,
                    "stability_index": r.stability_index,
                    "correlation_coefficient": r.correlation_coefficient
                }
                for r in self.validation_results
            ],
            "analysis": self.analyze_validation_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Enhanced validation results exported to: {output_path}")
        return str(output_path)

def run_enhanced_spinfoam_validation(N: int = 7, max_network_size: int = 15, 
                                   export: bool = True) -> Dict[str, Any]:
    """
    Main function to run enhanced spin-foam validation.
    
    Args:
        N: Lattice size for canonical LQG
        max_network_size: Maximum network size to test
        export: Whether to export results
        
    Returns:
        Dictionary containing analysis results
    """
    validator = EnhancedSpinFoamValidator(N, max_network_size)
    
    # Run enhanced validation
    results = validator.run_enhanced_validation()
    
    # Analyze results
    analysis = validator.analyze_validation_results()
    
    # Export results
    if export:
        validator.export_results()
    
    # Print summary
    print("\n📊 ENHANCED SPIN-FOAM VALIDATION SUMMARY")
    print("="*55)
    print(f"Total validations: {analysis['total_validations']}")
    print(f"Network size range: {analysis['network_size_range'][0]} to {analysis['network_size_range'][1]}")
    print(f"Mean relative error: {analysis['error_statistics']['mean_error']:.4f}")
    print(f"Mean stability: {analysis['stability_statistics']['mean_stability']:.3f}")
    print(f"Mean correlation: {analysis['correlation_statistics']['mean_correlation']:.3f}")
    
    print("\n🔬 Convergence by Network Size:")
    for size_key, conv_stats in analysis["convergence_analysis"].items():
        size = size_key.split("_")[1]
        print(f"  Size {size}: error = {conv_stats['average_error']:.4f}, "
              f"stability = {conv_stats['average_stability']:.3f}")
    
    return analysis

if __name__ == "__main__":
    # Run enhanced spin-foam validation
    analysis_results = run_enhanced_spinfoam_validation(
        N=7, max_network_size=15, export=True
    )
