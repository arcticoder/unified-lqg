#!/usr/bin/env python3
"""
LQR-1 LQG Fusion Reactor Technical Schematic Generator (Working Version)
Generates professional electrical/technical schematic using schemdraw

Requirements:
    pip install schemdraw matplotlib

Usage:
    python generate_lqr1_basic_schematic.py
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
    d += elm.Label().at((0, 8)).label('LQR-1 LQG Fusion Reactor Technical Schematic', fontsize=14)
    d += elm.Label().at((0, 7.5)).label('500 MW Thermal • 200 MW Electrical • LQG Polymer Enhancement', fontsize=12)
    
    # Main plasma chamber - central component
    d += elm.Rect(w=3, h=2, fill='lightcoral').at((0, 0))
    d += elm.Label().at((0, 0)).label('C1-C4\\nPLASMA CHAMBER\\n3.5m Major Radius\\nTungsten Construction')
    
    # Toroidal Field Coils (M1) - inductors around chamber
    d += elm.Inductor().left().at((-5, 0))
    d += elm.Label().at((-6, 0.5)).label('M1.1-M1.6\\nToroidal Field\\nYBCO Superconductor', fontsize=10)
    
    d += elm.Inductor().right().at((5, 0))
    d += elm.Label().at((6, 0.5)).label('M1.7-M1.12', fontsize=10)
    
    # Poloidal Field Coils (M2)
    d += elm.Inductor().up().at((-2, 3))
    d += elm.Label().at((-2, 4)).label('M2.1-M2.3\\nPoloidal Field\\nNbTi Superconductor', fontsize=10)
    
    d += elm.Inductor().up().at((2, 3))
    d += elm.Label().at((2, 4)).label('M2.4-M2.6', fontsize=10)
    
    # Power Supply System (PS1)
    d += elm.SourceSin().right().at((8, 0))
    d += elm.Label().at((9, 0.5)).label('PS1\\n50 MW Pulsed\\nThyristor-based', fontsize=10)
    
    # Protection systems (PS2, PS3)
    d += elm.Resistor().down().at((8, -2))
    d += elm.Label().at((8, -3)).label('PS2\\nDump Resistors\\n100 MJ', fontsize=10)
    
    d += elm.Rect(w=1.5, h=0.8).at((10, -2))
    d += elm.Label().at((10, -2)).label('PS3\\nQuench\\nDetection', fontsize=10)
    
    # LQG Polymer Field Generators (LQG1)
    d += elm.Antenna().at((-3, 4))
    d += elm.Label().at((-4, 4.5)).label('LQG1.1-LQG1.4\\nPolymer Field\\nsinc(πμ)', fontsize=10)
    
    d += elm.Antenna().at((3, 4))
    d += elm.Label().at((4, 4.5)).label('LQG1.5-LQG1.8', fontsize=10)
    
    d += elm.Antenna().at((-3, -4))
    d += elm.Label().at((-4, -4.5)).label('LQG1.9-LQG1.12', fontsize=10)
    
    d += elm.Antenna().at((3, -4))
    d += elm.Label().at((4, -4.5)).label('LQG1.13-LQG1.16', fontsize=10)
    
    # LQG Control Unit (LQG3)
    d += elm.Rect(w=2, h=1).at((0, 5))
    d += elm.Label().at((0, 5)).label('LQG3\\nCentral Control\\nβ(t) Dynamic', fontsize=10)
    
    # Neutral Beam Injectors (NBI1-NBI4) - using arrows
    d += elm.Arrow().at((-4, 0)).to((-1.5, 0))
    d += elm.Label().at((-4.5, 0.5)).label('NBI1\\n120 keV D⁰\\n40 MW', fontsize=10)
    
    d += elm.Arrow().at((4, 0)).to((1.5, 0))
    d += elm.Label().at((4.5, 0.5)).label('NBI2', fontsize=10)
    
    d += elm.Arrow().at((0, 3)).to((0, 1))
    d += elm.Label().at((0.5, 3.5)).label('NBI3', fontsize=10)
    
    d += elm.Arrow().at((0, -3)).to((0, -1))
    d += elm.Label().at((0.5, -3.5)).label('NBI4', fontsize=10)
    
    # Ion Sources for NBIs
    d += elm.SourceSin().left().at((-7, 0))
    d += elm.Label().at((-8, 0.5)).label('NBI2.1-NBI2.4\\nRF Ion Sources\\n50 A Current', fontsize=10)
    
    # Vacuum System (V1-V4)
    d += elm.Rect(w=1.5, h=1).at((-5, -2))
    d += elm.Label().at((-5, -2)).label('V1\\nTurbo Pump\\n2000 L/s', fontsize=10)
    
    d += elm.Rect(w=1.5, h=1).at((-5, -3.5))
    d += elm.Label().at((-5, -3.5)).label('V2\\nTi-Sub Pump\\n8000 L/s', fontsize=10)
    
    # Ion Gauge and RGA
    d += elm.Rect(w=1.2, h=0.8).at((-5, -5))
    d += elm.Label().at((-5, -5)).label('V3\\nIon Gauge\\n10⁻¹¹ Torr', fontsize=10)
    
    d += elm.Rect(w=1.2, h=0.8).at((-3, -5))
    d += elm.Label().at((-3, -5)).label('V4\\nRGA\\n1-200 amu', fontsize=10)
    
    # Cryogenic System (CR1-CR3)
    d += elm.Rect(w=2, h=1.2).at((-8, 3))
    d += elm.Label().at((-8, 3)).label('CR1\\nHe Refrigerator\\n1.8K Operation\\n500W @ 4.2K', fontsize=10)
    
    d += elm.Rect(w=1, h=0.6).at((-6, 4))
    d += elm.Label().at((-6, 4)).label('CR3\\nTemp\\nSensors', fontsize=10)
    
    # Control System (CS1-CS3)
    d += elm.Rect(w=2.5, h=1.5).at((6, 4))
    d += elm.Label().at((6, 4)).label('CS1\\nReal-time VME\\nPowerPC 1GHz', fontsize=10)
    
    # Plasma Diagnostics (PD1-PD3)
    d += elm.Rect(w=1.8, h=0.8).at((9, 3))
    d += elm.Label().at((9, 3)).label('PD1\\nThomson\\nScattering', fontsize=10)
    
    d += elm.Rect(w=1.8, h=0.8).at((9, 2))
    d += elm.Label().at((9, 2)).label('PD2\\nMagnetic\\nPickup', fontsize=10)
    
    d += elm.Rect(w=1.8, h=0.8).at((9, 1))
    d += elm.Label().at((9, 1)).label('PD3\\nBolometer\\nArray', fontsize=10)
    
    # Digital I/O modules
    d += elm.Rect(w=1.5, h=0.6).at((4, 5))
    d += elm.Label().at((4, 5)).label('CS2\\nDigital I/O\\n32-ch 50MHz', fontsize=10)
    
    d += elm.Rect(w=1.5, h=0.6).at((4, 4))
    d += elm.Label().at((4, 4)).label('CS3\\nAnalog Input\\n16-bit ±10V', fontsize=10)
    
    # Fuel Processing System (FP1-FP3)
    d += elm.Rect(w=2, h=1).at((6, -2))
    d += elm.Label().at((6, -2)).label('FP1\\nTritium Breeding\\nLi₄SiO₄ Modules', fontsize=10)
    
    d += elm.Rect(w=2, h=1).at((6, -4))
    d += elm.Label().at((6, -4)).label('FP2\\nTritium Recovery\\n1000 Ci/day', fontsize=10)
    
    d += elm.Rect(w=2, h=1).at((3, -4))
    d += elm.Label().at((3, -4)).label('FP3\\nPellet Injector\\n1000 Hz D-T', fontsize=10)
    
    # Safety Systems (S1-S3, ES1-ES3)
    d += elm.Rect(w=2, h=0.8).at((-8, -2))
    d += elm.Label().at((-8, -2)).label('S3\\nRadiation Monitor\\n24× Detectors', fontsize=10)
    
    d += elm.Rect(w=2, h=0.8).at((-5, -1))
    d += elm.Label().at((-5, -1)).label('ES1\\nEmergency\\nShutdown', fontsize=10)
    
    d += elm.Rect(w=2, h=0.8).at((-2, -1))
    d += elm.Label().at((-2, -1)).label('ES2\\nDisruption\\nMitigation', fontsize=10)
    
    d += elm.Rect(w=2, h=0.8).at((1, -1))
    d += elm.Label().at((1, -1)).label('ES3\\nFire\\nSuppression', fontsize=10)
    
    # Add connecting lines
    d += elm.Line().at((-7, 0)).to((-5, 0))  # Ion source to coil
    d += elm.Line().at((8, 0)).to((5, 0))    # Power supply to coil
    d += elm.Line().at((-4.5, -2)).to((-1.5, -1))  # Vacuum to chamber
    d += elm.Line().at((-7, 3)).to((-5, 0))  # Cryo to coil
    d += elm.Line().at((8, 3)).to((1.5, 0.5))   # Diagnostics to chamber
    d += elm.Line().at((5, -2)).to((1.5, -1))   # Fuel to chamber
    d += elm.Line().at((-5, -1)).to((8, -0.5))  # Emergency to power supply
    
    # Add specification text boxes
    d += elm.Label().at((-8, -6)).label('Power Distribution:', fontsize=12)
    d += elm.Label().at((-8, -6.5)).label('• 400 MW → LQG Drive Core', fontsize=10)
    d += elm.Label().at((-8, -7)).label('• 50 MW → Life Support', fontsize=10)
    d += elm.Label().at((-8, -7.5)).label('• 30 MW → Ship Systems', fontsize=10)
    d += elm.Label().at((-8, -8)).label('• 20 MW → Crew Support', fontsize=10)
    
    d += elm.Label().at((5, -6)).label('Performance Specifications:', fontsize=12)
    d += elm.Label().at((5, -6.5)).label('• H-factor = 1.94 (94% efficiency)', fontsize=10)
    d += elm.Label().at((5, -7)).label('• Te ≥ 15 keV, ne ≥ 10²⁰ m⁻³', fontsize=10)
    d += elm.Label().at((5, -7.5)).label('• τE ≥ 3.2 s confinement time', fontsize=10)
    d += elm.Label().at((5, -8)).label('• ≤10 mSv radiation exposure', fontsize=10)
    
    # Safety warning box
    d += elm.Rect(w=8, h=1.5, fill='mistyrose').at((0, -9.5))
    d += elm.Label().at((0, -9.5)).label('⚠️ BLACK AND RED LABEL SAFETY ⚠️\\nHigh Energy Plasma and Radiation Hazards\\nPersonnel require specialized training', fontsize=11)
    
    return d

def save_schematic():
    """Generate and save the LQR-1 schematic"""
    
    try:
        # Create the schematic
        schematic = create_lqr1_schematic()
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(18, 14))
        
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
