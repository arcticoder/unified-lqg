# Warp Predictive Framework

This repository implements a "Predictive Framework" stage for extracting laboratory-amenable predictions about microscopic wormhole stability, lifetimes, throat dynamics, and analogous outputs for analogue-gravity experiments. It is designed to sit downstream of PN/strong-curvature pipelines and consistency checks.

## LQG Integration & GPU Acceleration

The framework now includes Loop Quantum Gravity (LQG) integration and GPU-accelerated utilities:

### Quick Testing Commands

• **To generate a perfect-match JSON (GPU accelerated), run:**
  ```bash
  python generate_perfect_json_gpu.py
  ```
  If you have no matching (μ, ν), it will raise a RuntimeError.

• **To smoke-test every component on a tiny 3-site system:**
  ```bash
  python test_pipeline_quick.py
  ```

• **To run the full LQG-integrated pipeline:**
  ```bash
  python run_pipeline.py --use-quantum --lattice examples/example_reduced_variables.json
  ```

### New Example Files

• `examples/test_minimal_lattice.json` - Minimal 3-site system for smoke testing
• `examples/perfect_match_values_gpu.json` - GPU-generated perfect match results  
• `examples/perfect_match_values.json` - CPU-generated reference results

The smoke test (`test_pipeline_quick.py`) now verifies individual operator calls (e.g. `kin_space.kx_operator(site=0)`) so new contributors can catch basic import or signature errors early.

---

## Directory Structure
```

warp-predictive-framework/  
├── generate\_wormhole.py  
├── analyze\_stability.py  
├── compute\_lifetime.py  
├── map\_to\_analogue.py  
├── predictive\_config.am  
├── reference\_solutions/  
│ ├── morris\_thorne\_base.ndjson  
│ └── exotic\_matter\_eos.am  
├── outputs/  
│ ├── wormhole\_solutions.ndjson  
│ ├── stability\_spectrum.ndjson  
│ ├── lifetime\_estimates.ndjson  
│ └── analogue\_predictions.am  
└── examples/  
├── example\_wormhole\_input.am  
├── example\_stability\_output.ndjson  
├── example\_lifetime\_output.ndjson  
└── example\_analogue\_output.am

```
Each Python script reads from `predictive_config.am` and the appropriate reference files, then writes to `outputs/`. The `examples/` folder contains minimal toy inputs and outputs so you can validate the workflow without running heavy solvers.

## Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/arcticoder/warp-predictive-framework.git
    cd warp-predictive-framework
    ```

2.  **Install dependencies**  
    Create a virtual environment (optional) and install:
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    
*`requirements.txt` should list:*
    
```
numpy
scipy
sympy
python-ndjson
matplotlib # (optional; only if plotting is needed)
```
    
3.  **Populate `predictive_config.am`**  
Edit `predictive_config.am` to set global parameters. Example:
    
    ```asciimath
    [
        WormholeFamily     = "MorrisThorne",
        ThroatRadius       = 5e-35,
        ShapeFunction      = "b(r) = b0^2 / r",
        RedshiftFunction   = "Φ(r) = 0",
        ExoticEoS          = "phantom_scalar",
        CouplingConstants  = [ 1.0e-77 ],
        StabilityTolerance = 1e-6,
        LifetimeModel      = "HawkingBackreaction",
        LabAnalogueType    = "BEC_phonon",
        OutputLabels       = [
        "outputs/wormhole_solutions.ndjson",
        "outputs/stability_spectrum.ndjson",
        "outputs/lifetime_estimates.ndjson",
        "outputs/analogue_predictions.am"
        ]
    ]
    ```
    
4.  **Run each stage**
    
-   **Generate static wormhole solutions**
        
    ```bash
    python3 generate_wormhole.py \
        --config predictive_config.am \
        --out outputs/wormhole_solutions.ndjson
    ```
        
-   **Analyze linear stability**
        
    ```bash
    python3 analyze_stability.py \
        --input outputs/wormhole_solutions.ndjson \
        --config predictive_config.am \
        --out outputs/stability_spectrum.ndjson
    ```
        
-   **Compute lifetime estimates**
        
    ```bash
    python3 compute_lifetime.py \
        --input outputs/stability_spectrum.ndjson \
        --config predictive_config.am \
        --out outputs/lifetime_estimates.ndjson
    ```
        
-   **Map to analogue-gravity outputs**
        
    ```bash
    python3 map_to_analogue.py \
        --input outputs/lifetime_estimates.ndjson \
        --config predictive_config.am
    ```
        
    
After running `map_to_analogue.py`, you’ll find `outputs/analogue_predictions.am`.
    
5.  **Use the example folder**  
    To skip full solvers and test only the mapping step:
    
    ```bash
    cp examples/example_wormhole_input.am predictive_config.am
    cp examples/example_stability_output.ndjson outputs/stability_spectrum.ndjson
    cp examples/example_lifetime_output.ndjson outputs/lifetime_estimates.ndjson

    python3 map_to_analogue.py \
        --input outputs/lifetime_estimates.ndjson \
        --config predictive_config.am
    ```
    
Then inspect `examples/example_analogue_output.am` for expected analogue outputs.
    

## Physics/Implementation Notes

-   **Wormhole families**
    
    -   *Morris–Thorne throat:*  
        $b(r) = \frac{b_0^2}{r}, \quad \Phi(r) = 0.$
        
    -   To generalize, set `ShapeFunction = "b(r) = b0^n / r^{n-1}"` in `predictive_config.am`.
        
-   **Stability analysis**
    
    -   Solve the Sturm–Liouville eigenvalue problem for the perturbation $\\delta b$. Negative $\\omega^2$ indicates unstable modes.
        
    -   Write `"stable": true` if all eigenvalues > 0 up to `StabilityTolerance`.
        
-   **Lifetime/backreaction**
    
    -   Estimate growth timescale $\\tau = 1/|\\omega|$ for the dominant unstable mode.
        
    -   Semiclassical (Hawking-type) evaporation:  
        $\frac{d b_0}{dt} \approx -\,\kappa \,\frac{\hbar}{b_0^2},$  
        integrate from initial $b\_0$ to Planck length.
        
-   **Mapping to lab analogue**
    
    -   For a Bose–Einstein condensate, map Planck-unit $b\_0$ into an “acoustic horizon” radius $R\_{\\rm analog}$ using local sound speed $c\_s$ and healing length $\\xi$.
        
    -   Output in `analogue_predictions.am` as AsciiMath:
        
    ```asciimath
    [
        analogue_type       = "BEC_phonon",
        throat_radius_analog = "1e-6 m",
        instability_freq     = "2π × 1.2 kHz",
        lab_evap_time        = "0.02 s"
    ]
    ```
        

## Reference Solutions

-   **`reference_solutions/morris_thorne_base.ndjson`**
    
```json
{
    "label": "MT_static_b0=5e-35",
    "b0": 5e-35,
    "Phi_r": 0,
    "ValidRange": [5e-35, 20e-35]
}
```
    
-   **`reference_solutions/exotic_matter_eos.am`**
    
```asciimath
[ eos_type = "phantom_scalar", parameter = "w=-1.5" ]
```
    

## Examples Folder

-   `examples/example_wormhole_input.am`  
    Sets a toy throat radius (e.g. $b\_0=1\\times 10^{-35},\\mathrm{m}$) and a simple EoS.
    
-   `examples/example_stability_output.ndjson`  
    Contains one sample eigenmode record to bypass the eigenvalue solver.
    
-   `examples/example_lifetime_output.ndjson`  
    Lists idealized `"estimated_instability_time": 1e-42` and `"semi_classical_evap_time": 1e-36`.
    
-   `examples/example_analogue_output.am`  
    Shows the translated analogue-gravity predictions.

-   `examples/test_minimal_lattice.json` - Minimal 3-site system for smoke testing
-   `examples/perfect_match_values_gpu.json` - GPU-generated perfect match results  
-   `examples/perfect_match_values.json` - CPU-generated reference results
    

## Dependencies

-   **NumPy** (arrays, basic math)
    
-   **SciPy** (eigenvalue solvers, linear algebra)
    
-   **SymPy** (optional; symbolic curvature invariants or PN expansions)
    
-   **python-ndjson** (JSON-lines I/O)
    
-   **Matplotlib** (optional; plotting mode shapes or throat radius vs. time)
    

List these in a top-level `requirements.txt`:

```nginx
numpy
scipy
sympy
python-ndjson
matplotlib
```