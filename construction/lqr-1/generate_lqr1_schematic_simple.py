#!/usr/bin/env python3
"""
LQR-1 LQG Fusion Reactor Technical Schematic Generator (Simple Version)
Generates professional electrical/technical schematic using schemdraw

Requirements:
    pip install schemdraw matplotlib

Usage:
    python generate_lqr1_schematic_simple.py
"""

import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

def create_lqr1_schematic():
    """Generate LQR-1 technical schematic with proper electrical symbols"""
    
    # Create main schematic drawing
    d = schemdraw.Drawing()
    
    # Title
    d += elm.Label().label('LQR-1 LQG Fusion Reactor Technical Schematic').at((0, 8))
    d += elm.Label().label('500 MW Thermal • 200 MW Electrical • LQG Polymer Enhancement').at((0, 7.5))
    
    # Main plasma chamber - central component
    chamber = d.add(elm.Rect(w=3, h=2, fill='lightcoral'))
    d += elm.Label().at(chamber.center).label('C1-C4\\nPLASMA CHAMBER\\n3.5m Major Radius\\nTungsten Construction')
    
    # Toroidal Field Coils (M1) - inductors around chamber
    coil1 = d.add(elm.Inductor().left().at((-5, 0)).label('M1.1-M1.6\\nToroidal Field\\nYBCO Superconductor', loc='left'))
    coil2 = d.add(elm.Inductor().right().at((5, 0)).label('M1.7-M1.12', loc='right'))
    
    # Poloidal Field Coils (M2)
    pf1 = d.add(elm.Inductor().up().at((-2, 3)).label('M2.1-M2.3\\nPoloidal Field\\nNbTi Superconductor', loc='top'))
    pf2 = d.add(elm.Inductor().up().at((2, 3)).label('M2.4-M2.6', loc='top'))
    
    # Power Supply System (PS1)
    ps = d.add(elm.SourceSin().right().at((8, 0)).label('PS1\\n50 MW Pulsed\\nThyristor-based', loc='right'))
    
    # Connect power to coils
    d.add(elm.Line().at(ps.start).to(coil2.end))
    d.add(elm.Line().at(ps.start).to(pf2.end))
    
    # Protection systems (PS2, PS3)
    dump = d.add(elm.Resistor().down().at((8, -2)).label('PS2\\nDump Resistors\\n100 MJ', loc='bottom'))
    quench = d.add(elm.Rect(w=1.5, h=0.8).at((10, -2)).label('PS3\\nQuench\\nDetection'))
    
    # LQG Polymer Field Generators (LQG1)
    lqg1 = d.add(elm.Antenna().at((-3, 4)).label('LQG1.1-LQG1.4\\nPolymer Field\\nsinc(πμ)', loc='left'))
    lqg2 = d.add(elm.Antenna().at((3, 4)).label('LQG1.5-LQG1.8', loc='right'))
    lqg3 = d.add(elm.Antenna().at((-3, -4)).label('LQG1.9-LQG1.12', loc='left'))
    lqg4 = d.add(elm.Antenna().at((3, -4)).label('LQG1.13-LQG1.16', loc='right'))
    
    # LQG Control Unit (LQG3)
    lqg_ctrl = d.add(elm.Rect(w=2, h=1).at((0, 5)).label('LQG3\\nCentral Control\\nβ(t) Dynamic'))
    
    # Neutral Beam Injectors (NBI1-NBI4)
    d.add(elm.Arrow().at((-4, 0)).to((-1.5, 0)).label('NBI1\\n120 keV D⁰\\n40 MW', loc='left'))
    d.add(elm.Arrow().at((4, 0)).to((1.5, 0)).label('NBI2', loc='right'))
    d.add(elm.Arrow().at((0, 3)).to((0, 1)).label('NBI3', loc='top'))
    d.add(elm.Arrow().at((0, -3)).to((0, -1)).label('NBI4', loc='bottom'))
    
    # Ion Sources for NBIs
    ion_src = d.add(elm.SourceSin().left().at((-7, 0)).label('NBI2.1-NBI2.4\\nRF Ion Sources\\n50 A Current', loc='left'))
    d.add(elm.Line().at(ion_src.end).to(coil1.start))
    
    # Vacuum System (V1-V4)
    turbo = d.add(elm.Rect(w=1.5, h=1).at((-5, -2)).label('V1\\nTurbo Pump\\n2000 L/s', loc='left'))
    ti_sub = d.add(elm.Rect(w=1.5, h=1).at((-5, -3.5)).label('V2\\nTi-Sub Pump\\n8000 L/s', loc='left'))
    
    # Connect vacuum to chamber
    d.add(elm.Line().at(turbo.end).to((-1.5, -1)))
    d.add(elm.Line().at(ti_sub.end).to((-1.5, -1)))
    
    # Ion Gauge and RGA
    gauge = d.add(elm.Rect(w=1.2, h=0.8).at((-5, -5)).label('V3\\nIon Gauge\\n10⁻¹¹ Torr', loc='left'))
    rga = d.add(elm.Rect(w=1.2, h=0.8).at((-3, -5)).label('V4\\nRGA\\n1-200 amu', loc='bottom'))
    
    # Cryogenic System (CR1-CR3)
    cryo = d.add(elm.Rect(w=2, h=1.2).at((-8, 3)).label('CR1\\nHe Refrigerator\\n1.8K Operation\\n500W @ 4.2K', loc='left'))
    temp_sens = d.add(elm.Rect(w=1, h=0.6).at((-6, 4)).label('CR3\\nTemp\\nSensors', loc='top'))
    
    # Cryogenic connections (blue lines)
    d.add(elm.Line().at(cryo.end).to(coil1.start).color('blue').linewidth(3))
    d.add(elm.Line().at(cryo.end).to(pf1.start).color('blue').linewidth(3))
    
    # Control System (CS1-CS3)
    ctrl = d.add(elm.Rect(w=2.5, h=1.5).at((6, 4)).label('CS1\\nReal-time VME\\nPowerPC 1GHz', loc='top'))
    
    # Plasma Diagnostics (PD1-PD3)
    thomson = d.add(elm.Rect(w=1.8, h=0.8).at((9, 3)).label('PD1\\nThomson\\nScattering', loc='right'))
    pickup = d.add(elm.Rect(w=1.8, h=0.8).at((9, 2)).label('PD2\\nMagnetic\\nPickup', loc='right'))
    bolo = d.add(elm.Rect(w=1.8, h=0.8).at((9, 1)).label('PD3\\nBolometer\\nArray', loc='right'))
    
    # Diagnostic connections (dashed lines)
    d.add(elm.Line().at(thomson.start).to((1.5, 0.5)).linestyle('--'))
    d.add(elm.Line().at(pickup.start).to((1.5, 0)).linestyle('--'))
    d.add(elm.Line().at(bolo.start).to((1.5, -0.5)).linestyle('--'))
    
    # Digital I/O modules
    dio = d.add(elm.Rect(w=1.5, h=0.6).at((4, 5)).label('CS2\\nDigital I/O\\n32-ch 50MHz', loc='top'))
    analog = d.add(elm.Rect(w=1.5, h=0.6).at((4, 4)).label('CS3\\nAnalog Input\\n16-bit ±10V', loc='top'))
    
    # Fuel Processing System (FP1-FP3)
    breeding = d.add(elm.Rect(w=2, h=1).at((6, -2)).label('FP1\\nTritium Breeding\\nLi₄SiO₄ Modules', loc='bottom'))
    recovery = d.add(elm.Rect(w=2, h=1).at((6, -4)).label('FP2\\nTritium Recovery\\n1000 Ci/day', loc='bottom'))
    pellet = d.add(elm.Rect(w=2, h=1).at((3, -4)).label('FP3\\nPellet Injector\\n1000 Hz D-T', loc='bottom'))
    
    # Fuel connections
    d.add(elm.Line().at(breeding.start).to((1.5, -1)))
    d.add(elm.Line().at(pellet.end).to((0, -1)))
    
    # Safety Systems (S1-S3, ES1-ES3)
    rad_mon = d.add(elm.Rect(w=2, h=0.8).at((-8, -2)).label('S3\\nRadiation Monitor\\n24× Detectors', loc='left'))
    emerg = d.add(elm.Rect(w=2, h=0.8).at((-5, -1)).label('ES1\\nEmergency\\nShutdown', loc='bottom'))
    disrupt = d.add(elm.Rect(w=2, h=0.8).at((-2, -1)).label('ES2\\nDisruption\\nMitigation', loc='bottom'))
    fire = d.add(elm.Rect(w=2, h=0.8).at((1, -1)).label('ES3\\nFire\\nSuppression', loc='bottom'))
    
    # Safety connections
    d.add(elm.Line().at(emerg.end).to(ps.start).color('red').linewidth(2))
    d.add(elm.Line().at(rad_mon.end).to(ctrl.start).color('orange').linestyle('--'))
    
    return d

def save_schematic():
    """Generate and save the LQR-1 schematic"""
    
    try:
        # Create the schematic
        schematic = create_lqr1_schematic()
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw the schematic
        schematic.draw()
        
        # Save files
        plt.savefig('lqr-1_technical_schematic.png', dpi=300, bbox_inches='tight')
        plt.savefig('lqr-1_technical_schematic.svg', bbox_inches='tight')
        plt.close()
        
        print("✅ LQR-1 Technical Schematic Generated Successfully!")
        print("📁 Output files:")
        print("   • lqr-1_technical_schematic.png (300 DPI)")
        print("   • lqr-1_technical_schematic.svg (vector)")
        print()
        print("🔧 Component Summary:")
        print("   • C1-C4: Chamber Structure (tungsten construction)")
        print("   • M1: Toroidal Field Coils (18× YBCO superconductor)")
        print("   • M2: Poloidal Field Coils (6× NbTi superconductor)")
        print("   • LQG1-3: Polymer Field Generator Array (16× + control)")
        print("   • NBI1-4: Neutral Beam Injection System")
        print("   • V1-4: Vacuum System (turbo, Ti-sub, gauges)")
        print("   • CR1-3: Cryogenic System (1.8K operation)")
        print("   • PS1-3: Power Supply (50 MW + protection)")
        print("   • CS1-3: Control System (VME + diagnostics)")
        print("   • PD1-3: Plasma Diagnostics (Thomson, magnetic, bolometer)")
        print("   • FP1-3: Fuel Processing (breeding, recovery, injection)")
        print("   • S1-3/ES1-3: Safety Systems (monitoring + emergency)")
        
    except Exception as e:
        print(f"❌ Error generating schematic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    save_schematic()
