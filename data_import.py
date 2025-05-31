"""
Data Import Module for Warp Predictive Framework

This module handles importing data from the upstream warp-sensitivity-analysis repository,
including strong-curvature outputs, semi-classical predictions, and consistency check results.
"""

import os
import json
from pathlib import Path

# Path to the upstream warp-sensitivity-analysis repository
UPSTREAM_PATH = Path(__file__).parent.parent / "warp-sensitivity-analysis"

class DataImporter:
    """Handles importing data from upstream warp-sensitivity-analysis repository."""
    
    def __init__(self, upstream_path=None):
        self.upstream_path = Path(upstream_path) if upstream_path else UPSTREAM_PATH
        if not self.upstream_path.exists():
            raise FileNotFoundError(f"Upstream repository not found at: {self.upstream_path}")
    
    def load_strong_curvature_data(self):
        """Load strong curvature data (blackhole and cosmological)."""
        data = {}
        
        # Load blackhole curvature data (using actual filename)
        blackhole_path = self.upstream_path / "strong_curvature" / "blackhole_data.ndjson"
        if blackhole_path.exists():
            try:
                with open(blackhole_path) as f:
                    data['blackhole_curvature'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load blackhole data: {e}")
        
        # Load cosmological data
        cosmo_path = self.upstream_path / "strong_curvature" / "cosmo_data.ndjson"
        if cosmo_path.exists():
            try:
                with open(cosmo_path) as f:
                    data['cosmo_data'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load cosmo data: {e}")
        
        # Load unified strong models if available
        unified_path = self.upstream_path / "strong_curvature" / "unified_strong_models.ndjson"
        if unified_path.exists():
            try:
                with open(unified_path) as f:
                    data['unified_strong_models'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load unified models: {e}")
        
        return data
    
    def load_semi_classical_data(self):
        """Load semi-classical PN predictions and metadata."""
        data = {}
        
        semi_dir = self.upstream_path / "semi_classical"
        if not semi_dir.exists():
            return data
        
        # Load PN predictions
        pn_path = semi_dir / "pn_predictions.ndjson"
        if pn_path.exists():
            try:
                with open(pn_path) as f:
                    data['pn_predictions'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load PN predictions: {e}")
        
        # Load PN metadata
        pn_meta_path = semi_dir / "pn_metadata.am"
        if pn_meta_path.exists():
            data['pn_metadata'] = self._parse_am_file(pn_meta_path)
        
        return data
    
    def load_consistency_check_data(self):
        """Load consistency check outputs."""
        data = {}
        
        consistency_dir = self.upstream_path / "consistency_checks"
        if not consistency_dir.exists():
            return data
        
        # Load classical limit report
        classical_path = consistency_dir / "classical_limit_report.ndjson"
        if classical_path.exists():
            try:
                with open(classical_path) as f:
                    data['classical_limit_report'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load classical limit report: {e}")
        
        return data
    
    def find_planck_scale_candidates(self):
        """
        Analyze strong curvature data to find parameter sets that reach Planck-scale curvature.
        Returns candidates suitable for wormhole throat seeding.
        """
        strong_data = self.load_strong_curvature_data()
        candidates = []
        
        # Planck curvature scale (rough estimate: 1/l_planck^2)
        PLANCK_CURVATURE = 1e66  # m^-2
        
        if 'blackhole_curvature' in strong_data:
            for entry in strong_data['blackhole_curvature']:
                # Look for curvature indicators
                max_curvature = 0
                if 'max_kretschmann' in entry:
                    max_curvature = entry['max_kretschmann']
                elif 'kretschmann_peak' in entry:
                    max_curvature = entry['kretschmann_peak']
                elif 'curvature_scale' in entry:
                    max_curvature = entry['curvature_scale']
                
                if max_curvature > PLANCK_CURVATURE * 0.1:  # Within order of magnitude
                    candidates.append({
                        'type': 'blackhole',
                        'label': entry.get('label', 'unknown'),
                        'max_curvature': max_curvature,
                        'parameters': entry
                    })
        
        # Sort by maximum curvature (highest first)
        candidates.sort(key=lambda x: x['max_curvature'], reverse=True)
        return candidates
    
    def get_optimal_throat_parameters(self, candidate_type='auto'):
        """
        Extract optimal throat parameters from upstream data for wormhole generation.
        
        Args:
            candidate_type: 'auto', 'blackhole', 'cosmological', or specific label
        """
        candidates = self.find_planck_scale_candidates()
        
        if not candidates:
            # Fallback: try to extract from raw data
            strong_data = self.load_strong_curvature_data()
            
            if 'blackhole_curvature' in strong_data and strong_data['blackhole_curvature']:
                entry = strong_data['blackhole_curvature'][0]
                
                # Extract throat radius from mass or other parameters
                if 'mass' in entry:
                    mass = entry['mass']
                    G = 6.67e-11  # m^3 kg^-1 s^-2
                    c = 3e8       # m/s
                    throat_radius = 2 * G * mass / (c**2)
                elif 'r_min' in entry:
                    throat_radius = entry['r_min']
                else:
                    throat_radius = 1.6e-35  # Planck length
                
                return {
                    'throat_radius': throat_radius,
                    'shape_function': 'b0**2 / r',
                    'redshift_function': '0',
                    'source': entry.get('label', 'upstream_data'),
                    'background_type': 'blackhole'
                }
            
            # Final fallback to default parameters
            return {
                'throat_radius': 5e-35,  # meters (roughly Planck length)
                'shape_function': 'b0**2 / r',
                'redshift_function': '0',
                'source': 'default',
                'background_type': 'unknown'
            }
        
        # Select candidate
        if candidate_type == 'auto':
            selected = candidates[0]  # Highest curvature
        elif candidate_type in ['blackhole', 'cosmological']:
            selected = next((c for c in candidates if c['type'] == candidate_type), candidates[0])
        else:
            # Look for specific label
            selected = next((c for c in candidates if candidate_type in c['label']), candidates[0])
        
        # Extract throat parameters from selected candidate
        params = selected['parameters']
        
        # Estimate throat radius from characteristic scale
        if 'mass' in params:
            # For blackhole: throat ~ Schwarzschild radius
            mass = params['mass']
            G = 6.67e-11  # m^3 kg^-1 s^-2
            c = 3e8       # m/s
            throat_radius = 2 * G * mass / (c**2)
        elif 'r_min' in params:
            throat_radius = params['r_min']
        else:
            # Default based on curvature scale
            curvature = selected['max_curvature']
            throat_radius = (1 / curvature)**0.5 if curvature > 0 else 5e-35
        
        return {
            'throat_radius': throat_radius,
            'shape_function': 'b0**2 / r',  # Morris-Thorne default
            'redshift_function': '0',       # Static wormhole
            'source': selected['label'],
            'max_curvature': selected['max_curvature'],
            'background_type': selected['type']
        }
    
    def get_minimal_dataset(self):
        """
        Load only the minimal required dataset for standalone wormhole generation.
        This includes only the essential strong curvature data.
        """
        data = {}
        
        # Load essential blackhole curvature data (using actual filename)
        blackhole_path = self.upstream_path / "strong_curvature" / "blackhole_data.ndjson"
        if blackhole_path.exists():
            try:
                with open(blackhole_path) as f:
                    data['blackhole_curvature'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load blackhole data: {e}")
        else:
            print(f"Warning: Could not find {blackhole_path}")
        
        # Load essential cosmological data
        cosmo_path = self.upstream_path / "strong_curvature" / "cosmo_data.ndjson"
        if cosmo_path.exists():
            try:
                with open(cosmo_path) as f:
                    data['cosmo_data'] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                print(f"Warning: Could not load cosmo data: {e}")
        else:
            print(f"Warning: Could not find {cosmo_path}")
        
        return data
    
    def _parse_am_file(self, filepath):
        """Parse a simple AsciiMath config file."""
        try:
            with open(filepath) as f:
                contents = f.read().strip()
            # Simple parsing - replace with proper AsciiMath parser if needed
            if contents.startswith('[') and contents.endswith(']'):
                contents = contents[1:-1]
            return {'raw_content': contents, 'parsed': True}
        except Exception as e:
            return {'raw_content': '', 'parsed': False, 'error': str(e)}

def get_upstream_data(data_type='all'):
    """
    Convenience function to get upstream data.
    
    Args:
        data_type: 'strong', 'semi_classical', 'consistency', 'minimal', or 'all'
    """
    importer = DataImporter()
    
    if data_type == 'strong':
        return importer.load_strong_curvature_data()
    elif data_type == 'semi_classical':
        return importer.load_semi_classical_data()
    elif data_type == 'consistency':
        return importer.load_consistency_check_data()
    elif data_type == 'minimal':
        return importer.get_minimal_dataset()
    elif data_type == 'all':
        return {
            'strong_curvature': importer.load_strong_curvature_data(),
            'semi_classical': importer.load_semi_classical_data(),
            'consistency_checks': importer.load_consistency_check_data()
        }
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def load_for_wormhole_generation(background_type='auto'):
    """
    Convenience function specifically for wormhole generation.
    Loads minimal dataset and returns optimal throat parameters.
    
    Args:
        background_type: 'auto', 'blackhole', 'cosmological', or specific label
    """
    importer = DataImporter()
    
    # Load minimal dataset
    data = importer.get_minimal_dataset()
    
    # Get optimal parameters
    throat_params = importer.get_optimal_throat_parameters(background_type)
    
    return {
        'data': data,
        'throat_parameters': throat_params,
        'candidates': importer.find_planck_scale_candidates()
    }

if __name__ == "__main__":
    # Test the data import
    print("Testing data import from upstream repository...")
    
    try:
        importer = DataImporter()
        print(f"✓ Upstream path found: {importer.upstream_path}")
        
        # Test minimal dataset loading
        minimal_data = importer.get_minimal_dataset()
        print(f"✓ Minimal dataset keys: {list(minimal_data.keys())}")
        
        # Find Planck-scale candidates
        candidates = importer.find_planck_scale_candidates()
        print(f"✓ Found {len(candidates)} Planck-scale candidates")
        
        # Get optimal parameters
        optimal_params = importer.get_optimal_throat_parameters()
        print(f"✓ Optimal throat parameters: {optimal_params}")
        
        # Test convenience function
        wormhole_data = load_for_wormhole_generation()
        print(f"✓ Wormhole generation data loaded successfully")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Make sure the warp-sensitivity-analysis repository is in the parent directory")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
