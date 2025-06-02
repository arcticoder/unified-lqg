# Advanced LQG Framework Refinement Modules

This directory contains three advanced refinement modules designed to enhance the Loop Quantum Gravity framework with robust validation, benchmarking, and analysis capabilities.

## Modules Overview

### 1. Constraint Anomaly Scanner (`constraint_anomaly_scanner.py`)
- Tests the closure of constraint algebra across multiple lapse functions and parameters
- Features systematic scanning of Hamiltonian-Hamiltonian commutators
- Provides statistical analysis of anomaly rates
- Generates detailed visualization of closure errors

### 2. Continuum Benchmarking (`continuum_benchmarking.py`)
- Extends lattice refinement studies to larger N values 
- Implements Richardson extrapolation for continuum limit prediction
- Analyzes convergence rates and performance scaling
- Produces robust error estimates for extrapolated observables

### 3. Enhanced Spin-Foam Validation (`enhanced_spinfoam_validation.py`)
- Improves the cross-validation between canonical and spin-foam approaches
- Features energy scale normalization for consistent comparison
- Implements detailed mapping between spin values and canonical flux eigenvalues
- Provides comprehensive validation metrics and correlation analysis

## Usage

### Running All Refinements

You can run all three refinement modules through the demo script:

```bash
python advanced_refinements/demo_all_refinements.py
```

### Integration with Main Pipeline

To incorporate these advanced refinements into the main workflow, use the `--run-advanced-refinements` flag with the pipeline script:

```bash
python run_pipeline.py --run-advanced-refinements
```

You can also specify custom lattice sizes for testing:

```bash
python run_pipeline.py --run-advanced-refinements --lattice-sizes 3 5 7 9
```

To run the complete pipeline with quantum corrections and advanced refinements:

```bash
python run_pipeline.py --use-quantum --run-advanced-refinements
```

### VS Code Tasks

For convenience, the following tasks are available in VS Code:

1. **Run Advanced LQG Refinements** - Executes only the advanced refinement modules
2. **Run Complete LQG Pipeline with Refinements** - Runs the quantum-corrected pipeline and refinements

## Output and Visualization

All modules generate:
- Detailed logs of analysis steps
- JSON output files with comprehensive results
- Visualizations when appropriate (stored in outputs directory)
- Performance metrics and statistics

## Testing

Unit tests for the refinement modules are available in the tests directory:

```bash
python -m unittest tests/test_advanced_refinements.py
```
