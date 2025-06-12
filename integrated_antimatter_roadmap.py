#!/usr/bin/env python3
"""
Integrated Antimatter Production-to-Energy Roadmap
==================================================

Complete integration of production optimization and energy conversion
with realistic technology development pathways and economic analysis.

This module provides:
1. End-to-end system optimization from polymer corrections to electrical output
2. Technology readiness assessment and development timeline
3. Economic viability analysis with uncertainty quantification
4. Experimental validation pathways aligned with current facility capabilities
5. Risk assessment and mitigation strategies for commercial deployment

Integration Framework:
- Production: Enhanced Schwinger + Instanton rates with LQG corrections
- Capture: Magnetic bottle/Penning trap optimization
- Conversion: Material-optimized thermalization with advanced power cycles
- Control: Closed-loop energy management with UQ-based optimization
- Economics: Complete cost-benefit analysis with technology scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm, lognorm
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from production_ready_antimatter_roadmap import (
    ProductionReadyOptimizer, RealisticOptimizationConfig
)
from enhanced_matter_energy_conversion import (
    EnhancedConversionConfig, OptimizedConverter, AdvancedPowerCycles, 
    EnhancedMaterialDatabase
)

@dataclass
class IntegratedSystemConfig:
    """Configuration for complete integrated system"""
    # Production parameters
    target_electrical_power_W: float = 1000  # 1 kW target output
    min_viable_power_W: float = 100         # 100 W minimum for viability
    max_total_cost_USD: float = 500e6       # $500M maximum total investment
    
    # Technology development timeline
    development_timeline_years: float = 10   # 10-year development horizon
    technology_risk_tolerance: str = "medium"  # low, medium, high
    
    # Economic constraints
    electricity_price_USD_per_kWh: float = 0.20  # $0.20/kWh (high-value market)
    target_payback_years: float = 15        # 15-year payback acceptable
    required_irr_percent: float = 12        # 12% internal rate of return
    
    # Operational parameters
    facility_capacity_factor: float = 0.7   # 70% operational uptime
    facility_lifetime_years: float = 25     # 25-year operational life
    
    # Uncertainty quantification
    monte_carlo_samples: int = 10000        # UQ sample size
    confidence_level: float = 0.90          # 90% confidence intervals

class IntegratedSystemOptimizer:
    """
    Complete system optimizer integrating production through energy conversion
    """
    
    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        
        # Initialize sub-optimizers with consistent parameters
        self.production_config = RealisticOptimizationConfig(
            max_facility_cost_USD=config.max_total_cost_USD * 0.6,  # 60% for production
            target_cost_reduction=10.0  # Realistic 10x improvement
        )
        
        self.conversion_config = EnhancedConversionConfig(
            min_power_output=config.min_viable_power_W,
            target_efficiency=0.08,  # 8% total efficiency target
            max_system_cost=config.max_total_cost_USD * 0.4,  # 40% for conversion
            target_payback_years=config.target_payback_years
        )
        
        # Initialize optimizers
        self.production_optimizer = ProductionReadyOptimizer(self.production_config)
        self.converter = OptimizedConverter(self.conversion_config)
        self.power_cycles = AdvancedPowerCycles(self.conversion_config)
        
        print(f"🎯 Integrated Antimatter System Optimizer")
        print(f"   Target power output: {config.target_electrical_power_W} W")
        print(f"   Maximum investment: ${config.max_total_cost_USD/1e6:.0f}M")
        print(f"   Development timeline: {config.development_timeline_years} years")
        print(f"   Target payback: {config.target_payback_years} years")

    def find_optimal_production_rate(self) -> Dict:
        """
        Find optimal antimatter production rate for target electrical output
        """
        print(f"\n🔍 Finding Optimal Production Rate")
        
        def objective(log_production_rate):
            """Objective: minimize total cost for target power output"""
            production_rate = 10**log_production_rate[0]  # pairs/s
            
            try:
                # Quick production cost estimate
                E_field = 1e15  # V/m (representative high field)
                mu_g, b, S_inst = 0.2, 5.0, 1.0  # Representative optimal parameters
                
                # Production metrics
                prod_metrics = self.production_optimizer.calculate_production_cost_metrics(
                    E_field, mu_g, b, S_inst
                )
                
                if not prod_metrics['feasible']:
                    return 1e12
                
                # Conversion analysis
                captured_pairs_s = production_rate * self.conversion_config.capture_efficiency
                
                # Quick converter optimization for tungsten (fastest)
                conv_result = self.converter.optimize_converter_design('tungsten', captured_pairs_s)
                
                if not conv_result['feasible']:
                    return 1e12
                
                # Power cycle analysis (use best available)
                thermal_power = conv_result['net_thermal_power_W']
                hot_temp = conv_result['optimal_temperature_K']
                
                brayton = self.power_cycles.brayton_micro_turbine(thermal_power, hot_temp)
                teg = self.power_cycles.thermoelectric_generator(thermal_power, hot_temp)
                
                best_cycle = brayton if brayton['electrical_power_W'] > teg['electrical_power_W'] else teg
                
                if not best_cycle['feasible']:
                    return 1e12
                
                # Total system cost
                total_cost = (prod_metrics['total_cost_USD'] + 
                            conv_result['material_cost_USD'] + 
                            best_cycle['system_cost_USD'])
                
                # Penalty for not meeting target power
                power_ratio = best_cycle['electrical_power_W'] / self.config.target_electrical_power_W
                if power_ratio < 0.1:  # Less than 10% of target
                    return 1e12
                
                # Cost per watt as optimization metric
                cost_per_watt = total_cost / best_cycle['electrical_power_W']
                
                # Add penalty for deviating from target power
                power_penalty = abs(1 - power_ratio) * 1000
                
                return cost_per_watt + power_penalty
                
            except Exception as e:
                return 1e12
        
        # Optimize production rate (log scale from 1e2 to 1e6 pairs/s)
        bounds = [(2, 6)]  # log10(production_rate)
        result = differential_evolution(objective, bounds, seed=42, maxiter=50)
        
        optimal_log_rate = result.x[0]
        optimal_rate = 10**optimal_log_rate
        
        print(f"   Optimal production rate: {optimal_rate:.0e} pairs/s")
        print(f"   Optimization cost metric: {result.fun:.2f}")
        
        return {
            'optimal_production_rate_pairs_s': optimal_rate,
            'optimization_cost_metric': result.fun,
            'optimization_success': result.success
        }

    def execute_integrated_optimization(self) -> Dict:
        """
        Execute complete integrated system optimization
        """
        print(f"\n🚀 Integrated System Optimization")
        print("-" * 50)
        
        # Step 1: Find optimal production rate
        rate_optimization = self.find_optimal_production_rate()
        
        if not rate_optimization['optimization_success']:
            print("   ❌ Failed to find viable production rate")
            return {'status': 'failed', 'reason': 'No viable production rate found'}
        
        optimal_rate = rate_optimization['optimal_production_rate_pairs_s']
        
        # Step 2: Detailed production optimization
        print(f"\n📊 Detailed Production Analysis at {optimal_rate:.0e} pairs/s")
        production_results = self.production_optimizer.execute_comprehensive_optimization()
        
        if not production_results['best_result']:
            print("   ❌ No feasible production configurations")
            return {'status': 'failed', 'reason': 'No feasible production configurations'}
        
        best_production = production_results['best_result']
        
        # Step 3: Conversion system optimization
        print(f"\n🔄 Conversion System Optimization")
        captured_rate = optimal_rate * self.conversion_config.capture_efficiency
        
        # Test all materials
        converter_results = {}
        for material in self.conversion_config.converter_materials:
            result = self.converter.optimize_converter_design(material, captured_rate)
            if result['feasible']:
                converter_results[material] = result
        
        if not converter_results:
            print("   ❌ No feasible converter materials")
            return {'status': 'failed', 'reason': 'No feasible converter materials'}
        
        # Select best converter (minimum cost per watt)
        best_converter = min(converter_results.values(), key=lambda x: x['cost_per_watt_USD'])
        thermal_power = best_converter['net_thermal_power_W']
        hot_temp = best_converter['optimal_temperature_K']
        
        # Step 4: Power cycle optimization
        print(f"\n⚡ Power Cycle Analysis")
        
        cycles = {
            'brayton': self.power_cycles.brayton_micro_turbine(thermal_power, hot_temp),
            'thermoelectric': self.power_cycles.thermoelectric_generator(thermal_power, hot_temp),
            'stirling': self.power_cycles.stirling_micro_engine(thermal_power, hot_temp)
        }
        
        feasible_cycles = {k: v for k, v in cycles.items() if v['feasible']}
        
        if not feasible_cycles:
            print("   ❌ No feasible power cycles")
            return {'status': 'failed', 'reason': 'No feasible power cycles'}
        
        # Select best cycle (maximize electrical power)
        best_cycle = max(feasible_cycles.values(), key=lambda x: x['electrical_power_W'])
        
        # Step 5: Integrated system metrics
        total_cost = (best_production['total_cost_USD'] + 
                     best_converter['material_cost_USD'] + 
                     best_cycle['system_cost_USD'])
        
        electrical_power = best_cycle['electrical_power_W']
        total_efficiency = electrical_power / (optimal_rate * 2 * 0.511e6 * 1.602e-19)
        
        # Economic analysis
        annual_energy_kWh = (electrical_power * 8760 * self.config.facility_capacity_factor / 1000)
        annual_revenue = annual_energy_kWh * self.config.electricity_price_USD_per_kWh
        
        # Operating costs (maintenance + personnel)
        annual_operating_cost = total_cost * 0.05  # 5% of capital cost
        annual_profit = annual_revenue - annual_operating_cost
        
        # Financial metrics
        if annual_profit > 0:
            payback_years = total_cost / annual_profit
            npv_25_years = sum([annual_profit / (1.12)**year for year in range(1, 26)]) - total_cost
            irr = self.calculate_irr(total_cost, annual_profit, 25)
        else:
            payback_years = float('inf')
            npv_25_years = -total_cost
            irr = -100
        
        # Technology readiness assessment
        trl_assessment = self.assess_technology_readiness(best_production, best_converter, best_cycle)
        
        # Final integrated results
        integrated_results = {
            'status': 'success',
            'optimal_production_rate_pairs_s': optimal_rate,
            'best_production_config': best_production,
            'best_converter_config': best_converter,
            'best_power_cycle_config': best_cycle,
            'system_performance': {
                'electrical_power_W': electrical_power,
                'total_efficiency': total_efficiency,
                'thermal_power_W': thermal_power,
                'capture_efficiency': self.conversion_config.capture_efficiency
            },
            'economics': {
                'total_capital_cost_USD': total_cost,
                'annual_energy_kWh': annual_energy_kWh,
                'annual_revenue_USD': annual_revenue,
                'annual_operating_cost_USD': annual_operating_cost,
                'annual_profit_USD': annual_profit,
                'payback_years': payback_years,
                'npv_25_years_USD': npv_25_years,
                'irr_percent': irr
            },
            'technology_readiness': trl_assessment,
            'meets_targets': {
                'power_target': electrical_power >= self.config.target_electrical_power_W,
                'cost_target': total_cost <= self.config.max_total_cost_USD,
                'payback_target': payback_years <= self.config.target_payback_years,
                'irr_target': irr >= self.config.required_irr_percent
            }
        }
        
        # Print summary
        print(f"\n✅ INTEGRATED OPTIMIZATION COMPLETE")
        print(f"   Electrical output: {electrical_power:.0f} W")
        print(f"   Total efficiency: {total_efficiency:.1%}")
        print(f"   Total cost: ${total_cost/1e6:.1f}M")
        print(f"   Payback period: {payback_years:.1f} years")
        print(f"   NPV (25 years): ${npv_25_years/1e6:.1f}M")
        print(f"   IRR: {irr:.1f}%")
        print(f"   Technology readiness: {trl_assessment['overall_trl']}")
        
        # Check if targets are met
        targets_met = all(integrated_results['meets_targets'].values())
        print(f"   Targets met: {'✅ ALL' if targets_met else '⚠️ PARTIAL'}")
        
        return integrated_results

    def calculate_irr(self, initial_investment: float, annual_cash_flow: float, years: int) -> float:
        """Calculate internal rate of return"""
        if annual_cash_flow <= 0:
            return -100
        
        # Simple IRR approximation for constant cash flows
        irr_approx = (annual_cash_flow / initial_investment) * 100
        
        # More accurate calculation using Newton's method
        try:
            def npv_function(rate):
                return sum([annual_cash_flow / (1 + rate)**year for year in range(1, years + 1)]) - initial_investment
            
            # Newton's method
            rate = 0.10  # Initial guess: 10%
            for _ in range(20):
                npv = npv_function(rate)
                if abs(npv) < 1:
                    break
                
                # Numerical derivative
                h = 0.0001
                derivative = (npv_function(rate + h) - npv_function(rate - h)) / (2 * h)
                
                if abs(derivative) < 1e-10:
                    break
                    
                rate = rate - npv / derivative
                
                if rate < -0.99:  # Prevent negative rates below -99%
                    rate = -0.99
                    break
            
            return rate * 100
            
        except:
            return irr_approx

    def assess_technology_readiness(self, production: Dict, converter: Dict, cycle: Dict) -> Dict:
        """Assess overall technology readiness level"""
        
        # Individual component TRL assessment
        production_trl = production.get('technology_readiness_level', 3)
        
        # Converter TRL based on material and temperature
        if converter['material'] in ['tungsten', 'tantalum']:
            conv_trl = 6 if converter['optimal_temperature_K'] < 1200 else 4
        else:
            conv_trl = 4  # Advanced materials
        
        # Power cycle TRL
        cycle_type = cycle['cycle_type']
        if cycle_type == 'thermoelectric':
            cycle_trl = 7  # Mature technology
        elif cycle_type == 'brayton_micro':
            cycle_trl = 5  # Developing technology
        else:
            cycle_trl = 4  # Early development
        
        # Overall TRL (limited by lowest component)
        overall_trl = min(production_trl, conv_trl, cycle_trl)
        
        # Development timeline estimate
        trl_to_timeline = {
            1: 15, 2: 12, 3: 10, 4: 8, 5: 6, 6: 4, 7: 2, 8: 1, 9: 0
        }
        
        development_years = trl_to_timeline.get(overall_trl, 10)
        
        return {
            'production_trl': production_trl,
            'converter_trl': conv_trl,
            'power_cycle_trl': cycle_trl,
            'overall_trl': overall_trl,
            'estimated_development_years': development_years,
            'risk_level': 'low' if overall_trl >= 6 else 'medium' if overall_trl >= 4 else 'high'
        }

    def monte_carlo_uncertainty_analysis(self, base_results: Dict) -> Dict:
        """Perform Monte Carlo uncertainty quantification"""
        print(f"\n🎲 Monte Carlo Uncertainty Analysis")
        print(f"   Samples: {self.config.monte_carlo_samples:,}")
        
        # Define uncertainty parameters (standard deviations as fractions)
        uncertainties = {
            'production_rate': 0.30,      # 30% uncertainty in production rate
            'capture_efficiency': 0.20,   # 20% uncertainty in capture
            'conversion_efficiency': 0.15, # 15% uncertainty in conversion
            'power_cycle_efficiency': 0.10, # 10% uncertainty in power cycle
            'capital_cost': 0.25,         # 25% uncertainty in costs
            'operating_cost': 0.20,       # 20% uncertainty in operating costs
            'electricity_price': 0.30     # 30% uncertainty in electricity price
        }
        
        # Extract baseline values
        base_power = base_results['system_performance']['electrical_power_W']
        base_cost = base_results['economics']['total_capital_cost_USD']
        base_revenue = base_results['economics']['annual_revenue_USD']
        
        # Monte Carlo simulation
        power_samples = []
        npv_samples = []
        payback_samples = []
        
        for _ in range(self.config.monte_carlo_samples):
            # Sample uncertainty factors
            factors = {}
            for param, std in uncertainties.items():
                factors[param] = np.random.lognormal(0, std)
            
            # Calculate sample values
            sample_power = base_power * factors['production_rate'] * factors['capture_efficiency'] * factors['conversion_efficiency'] * factors['power_cycle_efficiency']
            sample_cost = base_cost * factors['capital_cost']
            sample_revenue = base_revenue * factors['electricity_price'] * (sample_power / base_power)
            sample_operating = sample_cost * 0.05 * factors['operating_cost']
            sample_profit = sample_revenue - sample_operating
            
            power_samples.append(sample_power)
            
            if sample_profit > 0:
                sample_payback = sample_cost / sample_profit
                payback_samples.append(min(sample_payback, 50))  # Cap at 50 years
                
                # NPV calculation
                npv = sum([sample_profit / (1.12)**year for year in range(1, 26)]) - sample_cost
                npv_samples.append(npv)
            else:
                payback_samples.append(50)  # Max payback
                npv_samples.append(-sample_cost)  # Negative NPV
        
        # Statistical analysis
        power_mean = np.mean(power_samples)
        power_std = np.std(power_samples)
        power_ci = np.percentile(power_samples, [5, 95])
        
        npv_mean = np.mean(npv_samples)
        npv_std = np.std(npv_samples)
        npv_ci = np.percentile(npv_samples, [5, 95])
        
        payback_mean = np.mean(payback_samples)
        payback_ci = np.percentile(payback_samples, [5, 95])
        
        # Success probabilities
        power_success_prob = np.mean([p >= self.config.min_viable_power_W for p in power_samples])
        npv_positive_prob = np.mean([npv > 0 for npv in npv_samples])
        payback_success_prob = np.mean([pb <= self.config.target_payback_years for pb in payback_samples])
        
        uncertainty_results = {
            'power_statistics': {
                'mean_W': power_mean,
                'std_W': power_std,
                'confidence_interval_90_W': power_ci.tolist(),
                'success_probability': power_success_prob
            },
            'npv_statistics': {
                'mean_USD': npv_mean,
                'std_USD': npv_std,
                'confidence_interval_90_USD': npv_ci.tolist(),
                'positive_probability': npv_positive_prob
            },
            'payback_statistics': {
                'mean_years': payback_mean,
                'confidence_interval_90_years': payback_ci.tolist(),
                'success_probability': payback_success_prob
            },
            'overall_success_probability': power_success_prob * npv_positive_prob * payback_success_prob
        }
        
        print(f"   Power output: {power_mean:.0f} ± {power_std:.0f} W")
        print(f"   NPV: ${npv_mean/1e6:.1f} ± {npv_std/1e6:.1f} M")
        print(f"   Success probability: {uncertainty_results['overall_success_probability']:.1%}")
        
        return uncertainty_results

def execute_complete_integrated_roadmap():
    """
    Execute the complete integrated antimatter production-to-energy roadmap
    """
    print("🚀 COMPLETE INTEGRATED ANTIMATTER PRODUCTION-TO-ENERGY ROADMAP")
    print("=" * 80)
    
    # Initialize configuration
    config = IntegratedSystemConfig()
    optimizer = IntegratedSystemOptimizer(config)
    
    # Execute integrated optimization
    results = optimizer.execute_integrated_optimization()
    
    if results['status'] != 'success':
        print(f"\n❌ OPTIMIZATION FAILED: {results['reason']}")
        return
    
    # Perform uncertainty analysis
    uncertainty_results = optimizer.monte_carlo_uncertainty_analysis(results)
    
    # Generate comprehensive report
    final_report = {
        'executive_summary': {
            'feasible': results['status'] == 'success',
            'electrical_power_W': results['system_performance']['electrical_power_W'],
            'total_cost_USD': results['economics']['total_capital_cost_USD'],
            'payback_years': results['economics']['payback_years'],
            'technology_readiness_level': results['technology_readiness']['overall_trl'],
            'development_timeline_years': results['technology_readiness']['estimated_development_years'],
            'overall_success_probability': uncertainty_results['overall_success_probability']
        },
        'detailed_results': results,
        'uncertainty_analysis': uncertainty_results,
        'recommendations': generate_recommendations(results, uncertainty_results),
        'implementation_roadmap': generate_implementation_roadmap(results)
    }
    
    # Save results
    output_dir = Path("integrated_antimatter_roadmap_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "complete_roadmap.json", 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Generate visualization
    generate_integrated_plots(results, uncertainty_results, output_dir)
    
    print(f"\n✅ COMPLETE INTEGRATED ROADMAP FINISHED")
    print(f"   Results saved to: {output_dir}")
    print(f"   Status: {'✅ FEASIBLE' if results['status'] == 'success' else '❌ INFEASIBLE'}")
    if results['status'] == 'success':
        print(f"   Power target achievement: {'✅' if results['meets_targets']['power_target'] else '❌'}")
        print(f"   Cost target achievement: {'✅' if results['meets_targets']['cost_target'] else '❌'}")
        print(f"   Overall success probability: {uncertainty_results['overall_success_probability']:.1%}")

def generate_recommendations(results: Dict, uncertainty: Dict) -> List[str]:
    """Generate specific recommendations based on results"""
    recommendations = []
    
    if results['meets_targets']['power_target']:
        recommendations.append("✅ Power target achieved - proceed with detailed design")
    else:
        recommendations.append("⚠️ Power target not met - consider higher production rates or efficiency improvements")
    
    if results['meets_targets']['cost_target']:
        recommendations.append("✅ Cost target achieved - economically viable")
    else:
        recommendations.append("⚠️ Cost target exceeded - focus on cost reduction strategies")
    
    if uncertainty['overall_success_probability'] > 0.7:
        recommendations.append("✅ High success probability - low technical risk")
    elif uncertainty['overall_success_probability'] > 0.3:
        recommendations.append("⚠️ Moderate success probability - implement risk mitigation")
    else:
        recommendations.append("❌ Low success probability - fundamental technology development needed")
    
    trl = results['technology_readiness']['overall_trl']
    if trl >= 6:
        recommendations.append("✅ High TRL - ready for prototype development")
    elif trl >= 4:
        recommendations.append("⚠️ Medium TRL - focused R&D program required")
    else:
        recommendations.append("❌ Low TRL - basic research and development needed")
    
    return recommendations

def generate_implementation_roadmap(results: Dict) -> Dict:
    """Generate detailed implementation roadmap with phases"""
    trl = results['technology_readiness']['overall_trl']
    dev_years = results['technology_readiness']['estimated_development_years']
    
    phases = {}
    
    if trl <= 3:
        phases['phase_1_basic_research'] = {
            'duration_years': min(4, dev_years * 0.4),
            'budget_fraction': 0.15,
            'objectives': ['Validate polymer corrections', 'Demonstrate enhanced rates', 'Material characterization']
        }
    
    if trl <= 5:
        phases['phase_2_technology_development'] = {
            'duration_years': min(4, dev_years * 0.4),
            'budget_fraction': 0.35,
            'objectives': ['Prototype components', 'System integration', 'Performance validation']
        }
    
    phases['phase_3_demonstration'] = {
        'duration_years': min(3, dev_years * 0.3),
        'budget_fraction': 0.30,
        'objectives': ['Build demonstration system', 'Validate economics', 'Optimize operations']
    }
    
    phases['phase_4_commercialization'] = {
        'duration_years': 2,
        'budget_fraction': 0.20,
        'objectives': ['Scale-up design', 'Commercial deployment', 'Market entry']
    }
    
    return {
        'phases': phases,
        'total_development_years': sum(p['duration_years'] for p in phases.values()),
        'total_budget_USD': results['economics']['total_capital_cost_USD']
    }

def generate_integrated_plots(results: Dict, uncertainty: Dict, output_dir: Path):
    """Generate comprehensive visualization plots"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Performance metrics
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['Power (W)', 'Efficiency (%)', 'Cost ($M)']
    values = [
        results['system_performance']['electrical_power_W'],
        results['system_performance']['total_efficiency'] * 100,
        results['economics']['total_capital_cost_USD'] / 1e6
    ]
    targets = [
        results['system_performance']['electrical_power_W'],  # Met or not based on color
        8,  # 8% efficiency target
        results['economics']['total_capital_cost_USD'] / 1e6
    ]
    
    bars = ax1.bar(metrics, values, alpha=0.7)
    for i, (bar, val, target) in enumerate(zip(bars, values, targets)):
        color = 'green' if val >= target * 0.8 else 'orange' if val >= target * 0.5 else 'red'
        bar.set_color(color)
    
    ax1.set_title('System Performance Metrics')
    ax1.set_ylabel('Value')
    
    # Uncertainty distributions
    ax2 = plt.subplot(2, 3, 2)
    power_samples = np.random.normal(
        uncertainty['power_statistics']['mean_W'],
        uncertainty['power_statistics']['std_W'],
        1000
    )
    ax2.hist(power_samples, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(uncertainty['power_statistics']['mean_W'], color='red', linestyle='--', label='Mean')
    ax2.set_title('Power Output Distribution')
    ax2.set_xlabel('Power (W)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Economics
    ax3 = plt.subplot(2, 3, 3)
    econ_labels = ['Revenue', 'OpCost', 'Profit']
    econ_values = [
        results['economics']['annual_revenue_USD'] / 1000,
        results['economics']['annual_operating_cost_USD'] / 1000,
        results['economics']['annual_profit_USD'] / 1000
    ]
    colors = ['green', 'red', 'blue']
    bars = ax3.bar(econ_labels, econ_values, color=colors, alpha=0.7)
    ax3.set_title('Annual Economics ($k)')
    ax3.set_ylabel('Thousands USD')
    
    # Technology readiness
    ax4 = plt.subplot(2, 3, 4)
    trl_components = ['Production', 'Converter', 'Power Cycle', 'Overall']
    trl_values = [
        results['technology_readiness']['production_trl'],
        results['technology_readiness']['converter_trl'],
        results['technology_readiness']['power_cycle_trl'],
        results['technology_readiness']['overall_trl']
    ]
    bars = ax4.bar(trl_components, trl_values, alpha=0.7)
    for bar, val in zip(bars, trl_values):
        color = 'green' if val >= 6 else 'orange' if val >= 4 else 'red'
        bar.set_color(color)
    
    ax4.set_title('Technology Readiness Levels')
    ax4.set_ylabel('TRL')
    ax4.set_ylim(0, 9)
    
    # Success probabilities
    ax5 = plt.subplot(2, 3, 5)
    prob_labels = ['Power', 'NPV+', 'Payback', 'Overall']
    prob_values = [
        uncertainty['power_statistics']['success_probability'],
        uncertainty['npv_statistics']['positive_probability'],
        uncertainty['payback_statistics']['success_probability'],
        uncertainty['overall_success_probability']
    ]
    bars = ax5.bar(prob_labels, prob_values, alpha=0.7)
    for bar, val in zip(bars, prob_values):
        color = 'green' if val >= 0.7 else 'orange' if val >= 0.3 else 'red'
        bar.set_color(color)
    
    ax5.set_title('Success Probabilities')
    ax5.set_ylabel('Probability')
    ax5.set_ylim(0, 1)
    
    # Target achievement
    ax6 = plt.subplot(2, 3, 6)
    target_labels = ['Power', 'Cost', 'Payback', 'IRR']
    target_achievement = [
        1 if results['meets_targets']['power_target'] else 0,
        1 if results['meets_targets']['cost_target'] else 0,
        1 if results['meets_targets']['payback_target'] else 0,
        1 if results['meets_targets']['irr_target'] else 0
    ]
    bars = ax6.bar(target_labels, target_achievement, alpha=0.7)
    for bar, achieved in zip(bars, target_achievement):
        bar.set_color('green' if achieved else 'red')
    
    ax6.set_title('Target Achievement')
    ax6.set_ylabel('Achieved (1=Yes, 0=No)')
    ax6.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "integrated_roadmap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    execute_complete_integrated_roadmap()
