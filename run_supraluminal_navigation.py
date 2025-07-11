#!/usr/bin/env python3
"""
Supraluminal Navigation System (48c Target) - Main Runner
=======================================================

Main execution script for the Supraluminal Navigation System.
Provides command-line interface for running navigation missions,
demonstrations, tests, and integration scenarios.

Usage:
  python run_supraluminal_navigation.py [command] [options]

Commands:
  demo        - Run navigation system demonstration
  mission     - Execute navigation mission with parameters
  test        - Run comprehensive test suite
  integrate   - Run integration demonstration
  benchmark   - Performance benchmarking
  config      - Generate or validate configuration

Author: GitHub Copilot
Date: July 11, 2025
Repository: unified-lqg
"""

import argparse
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.supraluminal_navigation import (
        SuperluminalNavigationSystem,
        NavigationTarget,
        demonstrate_supraluminal_navigation
    )
    from src.navigation_integration import (
        NavigationSystemIntegrator,
        demonstrate_integration
    )
    from tests.test_supraluminal_navigation import run_navigation_tests
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are properly installed")
    sys.exit(1)


def run_demo():
    """Run the navigation system demonstration"""
    print("🌌 Starting Supraluminal Navigation System Demonstration")
    print("=" * 60)
    
    try:
        nav_system = demonstrate_supraluminal_navigation()
        print(f"\n✅ Demonstration completed successfully!")
        return nav_system
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        return None


def run_mission(target_distance=4.24, target_velocity=48.0, duration_days=30.0):
    """Run a specific navigation mission"""
    print(f"🚀 Executing Navigation Mission")
    print(f"   Target distance: {target_distance} light-years")
    print(f"   Target velocity: {target_velocity}c")
    print(f"   Mission duration: {duration_days} days")
    print("=" * 50)
    
    try:
        # Initialize navigation system
        nav_system = SuperluminalNavigationSystem()
        
        # Define mission target
        target = NavigationTarget(
            target_position_ly=np.array([target_distance, 0.0, 0.0]),
            target_velocity_c=np.array([target_velocity, 0.0, 0.0]),
            mission_duration_days=duration_days,
            required_accuracy_ly=0.1
        )
        
        # Initialize mission
        print("📡 Initializing mission...")
        mission_init = nav_system.initialize_mission(target)
        print(f"   ✅ Mission initialized with {mission_init['stellar_detections']} stellar detections")
        
        # Run navigation updates
        print("🛸 Executing navigation updates...")
        for i in range(10):
            update = nav_system.update_navigation()
            if i % 3 == 0:  # Print every 3rd update
                print(f"   Step {i+1}: {update['system_status']}, "
                      f"Error: {update['position_error_ly']:.3f} ly, "
                      f"Velocity: {np.linalg.norm(nav_system.current_state.velocity_c):.1f}c")
            
            # Simulate acceleration
            if i < 5:
                nav_system.current_state.velocity_c += np.array([target_velocity/10, 0.0, 0.0])
        
        # Final status
        final_status = nav_system.get_navigation_status()
        print(f"\n📊 Mission Summary:")
        print(f"   Final status: {final_status['system_status']}")
        print(f"   Final velocity: {final_status['velocity_magnitude_c']:.1f}c")
        print(f"   Distance to target: {final_status['distance_to_target_ly']:.2f} ly")
        print(f"   Stellar detections: {final_status['stellar_detections']}")
        
        # Save mission results
        results_file = f"mission_results_{int(time.time())}.json"
        nav_system.save_navigation_state(results_file)
        print(f"   Results saved to: {results_file}")
        
        print(f"\n✅ Mission completed successfully!")
        return nav_system
        
    except Exception as e:
        print(f"\n❌ Mission failed: {e}")
        return None


def run_tests():
    """Run the comprehensive test suite"""
    print("🧪 Running Supraluminal Navigation Test Suite")
    print("=" * 50)
    
    try:
        success, results = run_navigation_tests()
        return success
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False


def run_integration():
    """Run integration demonstration"""
    print("🔗 Running Navigation System Integration Demonstration")
    print("=" * 55)
    
    try:
        integrator, results = demonstrate_integration()
        print(f"\n✅ Integration demonstration completed successfully!")
        return integrator, results
    except Exception as e:
        print(f"\n❌ Integration demonstration failed: {e}")
        return None, None


def run_benchmark():
    """Run performance benchmarking"""
    print("⚡ Running Navigation System Performance Benchmark")
    print("=" * 50)
    
    try:
        nav_system = SuperluminalNavigationSystem()
        
        # Benchmark navigation updates
        update_times = []
        print("📊 Benchmarking navigation updates...")
        
        # Setup benchmark mission
        target = NavigationTarget(
            target_position_ly=np.array([10.0, 0.0, 0.0]),
            target_velocity_c=np.array([48.0, 0.0, 0.0]),
            mission_duration_days=30.0
        )
        nav_system.initialize_mission(target)
        
        # Run benchmark
        for i in range(100):
            start_time = time.time()
            nav_system.update_navigation()
            end_time = time.time()
            update_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(update_times)
        min_time = np.min(update_times)
        max_time = np.max(update_times)
        std_time = np.std(update_times)
        
        print(f"\n📈 Benchmark Results (100 navigation updates):")
        print(f"   Average time: {avg_time:.2f} ms")
        print(f"   Minimum time: {min_time:.2f} ms")
        print(f"   Maximum time: {max_time:.2f} ms")
        print(f"   Standard deviation: {std_time:.2f} ms")
        print(f"   Update frequency: {1000/avg_time:.1f} Hz")
        
        # Performance assessment
        target_update_time = 100  # ms (10 Hz target)
        if avg_time <= target_update_time:
            print(f"   ✅ Performance: EXCELLENT (target: <{target_update_time}ms)")
        elif avg_time <= target_update_time * 2:
            print(f"   ⚠️  Performance: ACCEPTABLE (target: <{target_update_time}ms)")
        else:
            print(f"   ❌ Performance: NEEDS OPTIMIZATION (target: <{target_update_time}ms)")
        
        # Benchmark emergency deceleration
        print(f"\n🚨 Benchmarking emergency deceleration...")
        nav_system.current_state.velocity_c = np.array([48.0, 0.0, 0.0])
        
        start_time = time.time()
        emergency_result = nav_system.emergency_stop()
        end_time = time.time()
        
        emergency_time = (end_time - start_time) * 1000
        print(f"   Emergency response time: {emergency_time:.1f} ms")
        print(f"   Deceleration successful: {'✅' if emergency_result['reduction_result']['deceleration_successful'] else '❌'}")
        
        target_emergency_time = 1000  # ms (1 second target)
        if emergency_time <= target_emergency_time:
            print(f"   ✅ Emergency response: EXCELLENT (target: <{target_emergency_time}ms)")
        else:
            print(f"   ⚠️  Emergency response: NEEDS OPTIMIZATION (target: <{target_emergency_time}ms)")
        
        print(f"\n✅ Benchmark completed!")
        return {
            'update_stats': {
                'avg_ms': avg_time,
                'min_ms': min_time,
                'max_ms': max_time,
                'std_ms': std_time,
                'frequency_hz': 1000/avg_time
            },
            'emergency_response_ms': emergency_time
        }
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return None


def generate_config():
    """Generate or validate configuration file"""
    print("⚙️  Configuration Management")
    print("=" * 30)
    
    config_path = "config/supraluminal_navigation_config.json"
    
    if os.path.exists(config_path):
        print(f"📁 Existing configuration found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Configuration is valid JSON")
            
            # Validate required sections
            required_sections = [
                'mission_parameters',
                'gravimetric_sensor_array',
                'lensing_compensation',
                'course_correction',
                'emergency_protocols'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in config.get('supraluminal_navigation', {}):
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"⚠️  Missing configuration sections: {missing_sections}")
            else:
                print("✅ All required configuration sections present")
            
            return config
            
        except json.JSONDecodeError:
            print("❌ Configuration file contains invalid JSON")
            return None
    else:
        print(f"📝 Creating new configuration file: {config_path}")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Generate default configuration
        default_config = {
            "supraluminal_navigation": {
                "mission_parameters": {
                    "target_velocity_c": 48.0,
                    "maximum_velocity_c": 240.0,
                    "mission_duration_days": 30.0,
                    "navigation_accuracy_ly": 0.1
                },
                "gravimetric_sensor_array": {
                    "detection_range_ly": 10.0,
                    "stellar_mass_threshold_kg": 1e30,
                    "field_gradient_sensitivity_tesla_per_m": 1e-15
                },
                "emergency_protocols": {
                    "enabled": True,
                    "max_deceleration_g": 10.0,
                    "min_deceleration_time_s": 600,
                    "safety_margin": 1e12
                }
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"✅ Configuration file created successfully")
            return default_config
        except Exception as e:
            print(f"❌ Failed to create configuration file: {e}")
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Supraluminal Navigation System (48c Target)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_supraluminal_navigation.py demo
  python run_supraluminal_navigation.py mission --distance 4.24 --velocity 48
  python run_supraluminal_navigation.py test
  python run_supraluminal_navigation.py integrate
  python run_supraluminal_navigation.py benchmark
        """
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'mission', 'test', 'integrate', 'benchmark', 'config'],
        help='Command to execute'
    )
    
    # Mission-specific arguments
    parser.add_argument(
        '--distance',
        type=float,
        default=4.24,
        help='Target distance in light-years (default: 4.24 for Proxima Centauri)'
    )
    
    parser.add_argument(
        '--velocity',
        type=float,
        default=48.0,
        help='Target velocity in units of c (default: 48.0)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Mission duration in days (default: 30.0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🌌 Supraluminal Navigation System (48c Target)")
    print("=" * 50)
    print(f"Command: {args.command}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Execute command
    start_time = time.time()
    success = False
    
    try:
        if args.command == 'demo':
            result = run_demo()
            success = result is not None
            
        elif args.command == 'mission':
            result = run_mission(args.distance, args.velocity, args.duration)
            success = result is not None
            
        elif args.command == 'test':
            success = run_tests()
            
        elif args.command == 'integrate':
            integrator, results = run_integration()
            success = integrator is not None
            
        elif args.command == 'benchmark':
            result = run_benchmark()
            success = result is not None
            
        elif args.command == 'config':
            result = generate_config()
            success = result is not None
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Execution interrupted by user")
        success = False
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        success = False
    
    # Print execution summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n📊 Execution Summary:")
    print(f"   Command: {args.command}")
    print(f"   Duration: {execution_time:.2f} seconds")
    print(f"   Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
