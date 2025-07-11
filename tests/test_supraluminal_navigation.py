"""
Test Suite for Supraluminal Navigation System (48c Target)
========================================================

Comprehensive test suite validating all components of the supraluminal
navigation system including gravimetric sensing, lensing compensation,
course correction, and emergency protocols.

Author: GitHub Copilot
Date: July 11, 2025
Repository: unified-lqg
"""

import unittest
import numpy as np
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from supraluminal_navigation import (
        SuperluminalNavigationSystem,
        GravimetricNavigationArray,
        LensingCompensationSystem,
        SuperluminalCourseCorrector,
        EmergencyDecelerationController,
        NavigationState,
        NavigationTarget,
        StellarMass,
        DynamicBackreactionCalculator,
        MedicalGradeSafetySystem
    )
    from navigation_integration import NavigationSystemIntegrator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run tests from the unified-lqg directory")
    sys.exit(1)


class TestNavigationState(unittest.TestCase):
    """Test NavigationState data structure"""
    
    def test_navigation_state_creation(self):
        """Test creation of NavigationState"""
        state = NavigationState()
        
        self.assertEqual(len(state.position_ly), 3)
        self.assertEqual(len(state.velocity_c), 3)
        self.assertEqual(len(state.acceleration_c_per_s), 3)
        self.assertIsInstance(state.timestamp, float)
        self.assertEqual(state.backreaction_factor, 1.9443254780147017)
    
    def test_navigation_state_with_values(self):
        """Test NavigationState with specific values"""
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([48.0, 0.0, 0.0])
        
        state = NavigationState(
            position_ly=position,
            velocity_c=velocity,
            warp_field_strength=0.5
        )
        
        np.testing.assert_array_equal(state.position_ly, position)
        np.testing.assert_array_equal(state.velocity_c, velocity)
        self.assertEqual(state.warp_field_strength, 0.5)


class TestStellarMass(unittest.TestCase):
    """Test StellarMass data structure"""
    
    def test_stellar_mass_creation(self):
        """Test creation of StellarMass"""
        position = np.array([4.24, 0.0, 0.0])
        mass = 1.989e30  # Solar mass
        
        star = StellarMass(
            position_ly=position,
            mass_kg=mass,
            gravitational_signature=1e-45,
            detection_confidence=0.95,
            star_id="PROXIMA_CENTAURI"
        )
        
        np.testing.assert_array_equal(star.position_ly, position)
        self.assertEqual(star.mass_kg, mass)
        self.assertEqual(star.star_id, "PROXIMA_CENTAURI")


class TestGravimetricNavigationArray(unittest.TestCase):
    """Test gravimetric sensor array functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.array = GravimetricNavigationArray(detection_range_ly=10.0)
    
    def test_initialization(self):
        """Test array initialization"""
        self.assertEqual(self.array.detection_range_ly, 10.0)
        self.assertEqual(self.array.stellar_mass_threshold, 1e30)
        self.assertIsInstance(self.array.sensor_network, dict)
        self.assertEqual(len(self.array.detected_masses), 0)
    
    def test_graviton_detector_initialization(self):
        """Test graviton detector network initialization"""
        sensors = self.array.sensor_network
        
        self.assertIn('primary_array', sensors)
        self.assertIn('secondary_array', sensors)
        self.assertEqual(sensors['calibration_status'], 'OPERATIONAL')
        
        # Check primary array specifications
        primary = sensors['primary_array']
        self.assertEqual(primary['graviton_energy_range'], (1, 10))
        self.assertEqual(primary['signal_to_noise_ratio'], 15.1)
        self.assertEqual(primary['background_suppression'], 0.999)
    
    def test_gravitational_field_scanning(self):
        """Test gravitational field gradient scanning"""
        scan_results = self.array.scan_gravitational_field_gradients(5.0)
        
        self.assertIn('scan_volume_ly', scan_results)
        self.assertIn('gravitational_signatures', scan_results)
        self.assertIn('stellar_candidates', scan_results)
        self.assertEqual(scan_results['scan_volume_ly'], 5.0)
        self.assertIsInstance(scan_results['stellar_candidates'], list)
    
    def test_stellar_mass_detection(self):
        """Test stellar mass detection"""
        detected_masses = self.array.detect_stellar_masses(8.0)
        
        self.assertIsInstance(detected_masses, list)
        self.assertEqual(detected_masses, self.array.detected_masses)
        
        # Check detected masses are above threshold
        for mass in detected_masses:
            self.assertGreaterEqual(mass.mass_kg, self.array.stellar_mass_threshold)
            self.assertIsInstance(mass, StellarMass)
    
    def test_navigation_reference_validation(self):
        """Test validation of navigation references"""
        # Create test stellar masses
        test_masses = [
            StellarMass(
                position_ly=np.array([2.0, 0.0, 0.0]),
                mass_kg=2e30,  # Above threshold
                gravitational_signature=1e-45,
                detection_confidence=0.8,  # Above 0.6 threshold
                star_id="TEST_STAR_1"
            ),
            StellarMass(
                position_ly=np.array([15.0, 0.0, 0.0]),  # Outside range
                mass_kg=2e30,
                gravitational_signature=1e-47,
                detection_confidence=0.9,
                star_id="TEST_STAR_2"
            ),
            StellarMass(
                position_ly=np.array([1.0, 0.0, 0.0]),
                mass_kg=5e29,  # Below threshold
                gravitational_signature=1e-46,
                detection_confidence=0.7,
                star_id="TEST_STAR_3"
            )
        ]
        
        validated = self.array.validate_navigation_references(test_masses)
        
        # Only first star should pass validation
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].star_id, "TEST_STAR_1")


class TestLensingCompensationSystem(unittest.TestCase):
    """Test gravitational lensing compensation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compensator = LensingCompensationSystem()
        self.test_trajectory = NavigationState(
            position_ly=np.array([1.0, 0.0, 0.0]),
            velocity_c=np.array([48.0, 0.0, 0.0])
        )
        self.test_stellar_masses = [
            StellarMass(
                position_ly=np.array([2.0, 1.0, 0.0]),
                mass_kg=1.989e30,
                gravitational_signature=1e-45,
                detection_confidence=0.9,
                star_id="LENSING_STAR"
            )
        ]
    
    def test_initialization(self):
        """Test lensing compensator initialization"""
        self.assertIsNotNone(self.compensator.spacetime_geometry)
        self.assertIsNotNone(self.compensator.correction_algorithms)
    
    def test_spacetime_distortion_calculation(self):
        """Test spacetime distortion calculation"""
        distortion = self.compensator.calculate_spacetime_distortion(
            self.test_stellar_masses, self.test_trajectory
        )
        
        self.assertIn('total_distortion', distortion)
        self.assertIn('components', distortion)
        self.assertIn('max_component_magnitude', distortion)
        
        # Check distortion vector dimensions
        total_distortion = distortion['total_distortion']
        self.assertEqual(len(total_distortion), 3)
        
        # Check component information
        components = distortion['components']
        self.assertEqual(len(components), len(self.test_stellar_masses))
        self.assertEqual(components[0]['star_id'], "LENSING_STAR")
    
    def test_metric_adjustments(self):
        """Test metric adjustment calculations"""
        distortion = self.compensator.calculate_spacetime_distortion(
            self.test_stellar_masses, self.test_trajectory
        )
        
        adjustments = self.compensator.compute_metric_adjustments(
            distortion, self.test_trajectory
        )
        
        self.assertIn('metric_correction', adjustments)
        self.assertIn('relativistic_factor', adjustments)
        self.assertIn('correction_magnitude', adjustments)
        
        # Check relativistic factor is reasonable
        self.assertGreater(adjustments['relativistic_factor'], 1.0)
    
    def test_warp_field_geometry_adjustment(self):
        """Test warp field geometry adjustments"""
        distortion = self.compensator.calculate_spacetime_distortion(
            self.test_stellar_masses, self.test_trajectory
        )
        adjustments = self.compensator.compute_metric_adjustments(
            distortion, self.test_trajectory
        )
        
        geometry_result = self.compensator.adjust_warp_field_geometry(
            self.test_trajectory, adjustments
        )
        
        self.assertIn('position_correction', geometry_result)
        self.assertIn('corrected_position', geometry_result)
        self.assertIn('corrected_velocity', geometry_result)
        self.assertIn('new_backreaction_factor', geometry_result)
        self.assertTrue(geometry_result['correction_applied'])


class TestSuperLuminalCourseCorrector(unittest.TestCase):
    """Test supraluminal course correction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.corrector = SuperluminalCourseCorrector()
        self.current_state = NavigationState(
            position_ly=np.array([1.0, 0.0, 0.0]),
            velocity_c=np.array([40.0, 0.0, 0.0])
        )
        self.target = NavigationTarget(
            target_position_ly=np.array([4.24, 0.0, 0.0]),
            target_velocity_c=np.array([48.0, 0.0, 0.0]),
            mission_duration_days=30.0
        )
    
    def test_initialization(self):
        """Test course corrector initialization"""
        self.assertIsNotNone(self.corrector.dynamic_beta)
        self.assertIsNotNone(self.corrector.navigation_optimizer)
    
    def test_course_correction_execution(self):
        """Test course correction execution"""
        correction_result = self.corrector.execute_course_correction(
            self.current_state, self.target
        )
        
        self.assertIn('beta_optimized', correction_result)
        self.assertIn('warp_adjustment', correction_result)
        self.assertIn('correction_result', correction_result)
        
        # Check beta optimization
        beta = correction_result['beta_optimized']
        self.assertGreater(beta, 0.5)
        self.assertLess(beta, 3.0)
    
    def test_trajectory_correction_computation(self):
        """Test trajectory correction computation"""
        beta_optimized = 2.0
        adjustment = self.corrector.compute_trajectory_correction(
            beta_optimized, self.current_state, self.target
        )
        
        self.assertIn('position_error', adjustment)
        self.assertIn('velocity_error', adjustment)
        self.assertIn('position_correction', adjustment)
        self.assertIn('velocity_correction', adjustment)
        self.assertIn('required_acceleration', adjustment)
        
        # Check error calculations
        position_error = adjustment['position_error']
        expected_error = self.target.target_position_ly - self.current_state.position_ly
        np.testing.assert_array_equal(position_error, expected_error)
    
    def test_navigation_correction_application(self):
        """Test application of navigation corrections"""
        warp_adjustment = {
            'position_correction': np.array([0.1, 0.0, 0.0]),
            'velocity_correction': np.array([8.0, 0.0, 0.0]),
            'required_acceleration': np.array([0.01, 0.0, 0.0]),
            'beta_factor': 2.0
        }
        
        result = self.corrector.apply_navigation_correction(
            self.current_state, warp_adjustment
        )
        
        self.assertIn('updated_state', result)
        self.assertIn('correction_applied', result)
        self.assertTrue(result['correction_applied'])
        
        # Check velocity limiting
        updated_state = result['updated_state']
        velocity_magnitude = np.linalg.norm(updated_state.velocity_c)
        self.assertLessEqual(velocity_magnitude, 240)  # Should be limited to 240c


class TestDynamicBackreactionCalculator(unittest.TestCase):
    """Test dynamic backreaction factor calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = DynamicBackreactionCalculator()
    
    def test_initialization(self):
        """Test calculator initialization"""
        self.assertEqual(self.calculator.base_beta, 1.9443254780147017)
    
    def test_navigation_beta_calculation(self):
        """Test navigation beta calculation"""
        # Test at rest
        state_rest = NavigationState()
        beta_rest = self.calculator.calculate_navigation_beta(state_rest)
        self.assertAlmostEqual(beta_rest, self.calculator.base_beta, places=6)
        
        # Test at 48c
        state_48c = NavigationState(velocity_c=np.array([48.0, 0.0, 0.0]))
        beta_48c = self.calculator.calculate_navigation_beta(state_48c)
        self.assertGreater(beta_48c, beta_rest)
        
        # Test with acceleration
        state_accel = NavigationState(
            velocity_c=np.array([48.0, 0.0, 0.0]),
            acceleration_c_per_s=np.array([1.0, 0.0, 0.0])
        )
        beta_accel = self.calculator.calculate_navigation_beta(state_accel)
        self.assertGreater(beta_accel, beta_48c)
    
    def test_beta_stability_limits(self):
        """Test beta factor stability limits"""
        # Test extreme velocity
        extreme_state = NavigationState(velocity_c=np.array([500.0, 0.0, 0.0]))
        beta_extreme = self.calculator.calculate_navigation_beta(extreme_state)
        
        # Should be within stability limits
        self.assertGreaterEqual(beta_extreme, 0.5)
        self.assertLessEqual(beta_extreme, 3.0)


class TestEmergencyDecelerationController(unittest.TestCase):
    """Test emergency deceleration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = EmergencyDecelerationController()
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.max_deceleration_g, 10.0)
        self.assertEqual(self.controller.min_deceleration_time_s, 600)
        self.assertEqual(self.controller.safety_margin, 10**12)
    
    def test_safe_deceleration_curve_calculation(self):
        """Test safe deceleration curve calculation"""
        profile = self.controller.calculate_safe_deceleration_curve(48.0, 1.0)
        
        self.assertIn('time_points', profile)
        self.assertIn('velocity_profile', profile)
        self.assertIn('acceleration_profile', profile)
        self.assertIn('total_time_s', profile)
        self.assertIn('safety_compliant', profile)
        
        # Check time requirement
        self.assertGreaterEqual(profile['total_time_s'], self.controller.min_deceleration_time_s)
        
        # Check velocity profile
        velocities = profile['velocity_profile']
        self.assertAlmostEqual(velocities[0], 48.0, places=1)  # Start at 48c
        self.assertAlmostEqual(velocities[-1], 1.0, places=1)  # End at 1c
        
        # Check safety compliance
        self.assertTrue(profile['safety_compliant'])
    
    def test_emergency_deceleration_execution(self):
        """Test emergency deceleration execution"""
        result = self.controller.execute_emergency_deceleration(48.0, 0.1)
        
        self.assertIn('deceleration_profile', result)
        self.assertIn('safety_constraints', result)
        self.assertIn('reduction_result', result)
        
        # Check deceleration success
        reduction = result['reduction_result']
        self.assertTrue(reduction['deceleration_successful'])
        self.assertLess(reduction['final_velocity_c'], 48.0)
    
    def test_controlled_velocity_reduction(self):
        """Test controlled velocity reduction"""
        # Create test deceleration profile
        profile = self.controller.calculate_safe_deceleration_curve(48.0, 1.0)
        safety_constraints = {'margin_factor': 1e12}
        
        result = self.controller.controlled_velocity_reduction(profile, safety_constraints)
        
        self.assertIn('execution_log', result)
        self.assertIn('final_velocity_c', result)
        self.assertIn('deceleration_successful', result)
        
        # Check execution log
        log = result['execution_log']
        self.assertGreater(len(log), 0)
        
        # Check constraint monitoring in log entries
        for entry in log[:5]:  # Check first 5 entries
            self.assertIn('constraint_status', entry)
            self.assertIn('stability_status', entry)


class TestMedicalGradeSafetySystem(unittest.TestCase):
    """Test medical-grade safety system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.safety_system = MedicalGradeSafetySystem()
    
    def test_initialization(self):
        """Test safety system initialization"""
        self.assertEqual(self.safety_system.safety_margin, 10**12)
        self.assertIn('max_acceleration_g', self.safety_system.who_biological_limits)
    
    def test_biological_limits_enforcement(self):
        """Test biological limits enforcement"""
        # Test compliant profile
        compliant_profile = {'max_acceleration_g': 1e-11}  # Well below limit
        result_compliant = self.safety_system.enforce_biological_limits(compliant_profile)
        
        self.assertTrue(result_compliant['compliant'])
        self.assertGreater(result_compliant['margin_factor'], 1.0)
        
        # Test non-compliant profile
        non_compliant_profile = {'max_acceleration_g': 20.0}  # Above WHO limit
        result_non_compliant = self.safety_system.enforce_biological_limits(non_compliant_profile)
        
        self.assertFalse(result_non_compliant['compliant'])
        self.assertLess(result_non_compliant['margin_factor'], 1.0)


class TestSupraluminalNavigationSystem(unittest.TestCase):
    """Test complete supraluminal navigation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.nav_system = SuperluminalNavigationSystem()
        self.test_target = NavigationTarget(
            target_position_ly=np.array([4.24, 0.0, 0.0]),
            target_velocity_c=np.array([48.0, 0.0, 0.0]),
            mission_duration_days=30.0
        )
    
    def test_initialization(self):
        """Test navigation system initialization"""
        self.assertEqual(self.nav_system.system_status, 'INITIALIZED')
        self.assertIsNotNone(self.nav_system.gravimetric_array)
        self.assertIsNotNone(self.nav_system.lensing_compensator)
        self.assertIsNotNone(self.nav_system.course_corrector)
        self.assertIsNotNone(self.nav_system.emergency_controller)
    
    def test_mission_initialization(self):
        """Test mission initialization"""
        result = self.nav_system.initialize_mission(self.test_target)
        
        self.assertIn('mission_target', result)
        self.assertIn('stellar_detections', result)
        self.assertIn('initial_solution', result)
        self.assertEqual(self.nav_system.system_status, 'MISSION_INITIALIZED')
        self.assertEqual(self.nav_system.target, self.test_target)
    
    def test_navigation_updates(self):
        """Test navigation update cycle"""
        # Initialize mission first
        self.nav_system.initialize_mission(self.test_target)
        
        # Perform navigation update
        update_result = self.nav_system.update_navigation()
        
        self.assertIn('stellar_detections', update_result)
        self.assertIn('lensing_compensation', update_result)
        self.assertIn('course_correction', update_result)
        self.assertIn('position_error_ly', update_result)
        self.assertIn('system_status', update_result)
        
        # Check update performance
        self.assertLess(update_result['update_duration_ms'], 1000)  # Should complete in <1s
    
    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        # Set initial velocity
        self.nav_system.current_state.velocity_c = np.array([48.0, 0.0, 0.0])
        
        emergency_result = self.nav_system.emergency_stop()
        
        self.assertIn('reduction_result', emergency_result)
        self.assertEqual(self.nav_system.system_status, 'EMERGENCY_STOPPED')
        
        # Check final velocity is reduced
        final_velocity = np.linalg.norm(self.nav_system.current_state.velocity_c)
        self.assertLess(final_velocity, 48.0)
    
    def test_navigation_status(self):
        """Test navigation status reporting"""
        status = self.nav_system.get_navigation_status()
        
        self.assertIn('system_status', status)
        self.assertIn('current_position_ly', status)
        self.assertIn('current_velocity_c', status)
        self.assertIn('velocity_magnitude_c', status)
        self.assertIn('stellar_detections', status)
        self.assertIn('backreaction_factor', status)
    
    def test_navigation_state_persistence(self):
        """Test saving and loading navigation state"""
        # Initialize mission
        self.nav_system.initialize_mission(self.test_target)
        
        # Save state to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            success = self.nav_system.save_navigation_state(temp_filename)
            self.assertTrue(success)
            
            # Verify file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                saved_data = json.load(f)
            
            # Check required fields
            self.assertIn('navigation_state', saved_data)
            self.assertIn('target', saved_data)
            self.assertIn('stellar_map', saved_data)
            self.assertIn('system_status', saved_data)
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestNavigationSystemIntegrator(unittest.TestCase):
    """Test navigation system integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integrator = NavigationSystemIntegrator()
    
    def test_initialization(self):
        """Test integrator initialization"""
        self.assertIsNotNone(self.integrator.config)
        self.assertIsNotNone(self.integrator.repo_paths)
        self.assertIn('energy', self.integrator.repo_paths)
    
    def test_config_loading(self):
        """Test configuration loading"""
        # Test with default config
        config = self.integrator._default_config()
        self.assertIn('mission_parameters', config)
        self.assertIn('gravimetric_sensor_array', config)
    
    def test_repository_integration_check(self):
        """Test repository integration checking"""
        integration_status = self.integrator.check_repository_integrations()
        
        self.assertIsInstance(integration_status, dict)
        
        # Check expected repositories
        expected_repos = ['energy', 'warp_spacetime_stability', 'artificial_gravity', 
                         'medical_tractor', 'warp_field_coils']
        for repo in expected_repos:
            self.assertIn(repo, integration_status)
            self.assertIn('available', integration_status[repo])
            self.assertIn('path', integration_status[repo])
    
    def test_navigation_system_initialization(self):
        """Test navigation system initialization through integrator"""
        nav_system = self.integrator.initialize_navigation_system()
        
        self.assertIsInstance(nav_system, SuperluminalNavigationSystem)
        self.assertEqual(self.integrator.navigation_system, nav_system)
    
    def test_integration_simulations(self):
        """Test integration simulation methods"""
        # Create test data
        test_masses = [
            StellarMass(
                position_ly=np.array([2.0, 0.0, 0.0]),
                mass_kg=2e30,
                gravitational_signature=1e-45,
                detection_confidence=0.9,
                star_id="TEST_STAR"
            )
        ]
        
        test_state = NavigationState(velocity_c=np.array([48.0, 0.0, 0.0]))
        
        # Test graviton detection simulation
        graviton_result = self.integrator.simulate_graviton_detection_integration(test_masses)
        self.assertIn('graviton_detections', graviton_result)
        self.assertIn('integration_status', graviton_result)
        
        # Test stability monitoring simulation
        stability_result = self.integrator.simulate_stability_monitoring_integration(test_state)
        self.assertIn('stability_metrics', stability_result)
        self.assertIn('monitoring_active', stability_result)
        
        # Test field generation simulation
        field_result = self.integrator.simulate_field_generation_integration(test_state)
        self.assertIn('field_parameters', field_result)
        self.assertIn('coordination_active', field_result)
    
    def test_integration_report_generation(self):
        """Test integration report generation"""
        report = self.integrator.generate_integration_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('Supraluminal Navigation System', report)
        self.assertIn('Repository Integrations', report)
        self.assertIn('Navigation Capabilities', report)


def run_navigation_tests():
    """Run all navigation system tests"""
    print("🧪 Running Supraluminal Navigation System Test Suite")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestNavigationState,
        TestStellarMass,
        TestGravimetricNavigationArray,
        TestLensingCompensationSystem,
        TestSuperLuminalCourseCorrector,
        TestDynamicBackreactionCalculator,
        TestEmergencyDecelerationController,
        TestMedicalGradeSafetySystem,
        TestSupraluminalNavigationSystem,
        TestNavigationSystemIntegrator
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ All tests passed! Supraluminal Navigation System is ready for deployment! 🚀")
    else:
        print(f"\n⚠️  Some tests failed. Please review and fix issues before deployment.")
    
    return success, result


if __name__ == "__main__":
    success, test_result = run_navigation_tests()
