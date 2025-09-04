"""
Test script to validate the standalone testing workflow
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_standalone_testing():
    """Test the complete standalone testing workflow"""
    print("🧪 Testing Standalone Testing Workflow")
    print("=" * 50)
    
    try:
        # Test 1: Import all required modules
        print("1. Testing imports...")
        from vestim.gui.src.test_selection_gui_qt import TestSelectionGUI
        from vestim.gui.src.standalone_testing_gui_qt import VEstimStandaloneTestingGUI
        from vestim.gateway.src.standalone_testing_manager_qt import VEstimStandaloneTestingManager
        from vestim.gui.src.standalone_augmentation_gui_qt import StandaloneAugmentationGUI
        print("   ✅ All imports successful")
        
        # Test 2: Check if required directories exist
        print("2. Checking for job directories...")
        output_dir = os.path.join(project_root, 'output')
        job_folders = []
        job_path = None
        
        if os.path.exists(output_dir):
            job_folders = [f for f in os.listdir(output_dir) if f.startswith('job_')]
            if job_folders:
                latest_job = max(job_folders)
                job_path = os.path.join(output_dir, latest_job)
                print(f"   ✅ Found job directory: {latest_job}")
            else:
                print("   ❌ No job folders found in output directory")
        else:
            print("   ❌ Output directory not found")
            
        # Test 3: Check job structure...
        print("3. Checking job structure...")
        if job_folders:
            # Check for required files
            required_files = ['hyperparams.json', 'job_metadata.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(job_path, f))]
            if not missing_files:
                print("   ✅ Required job files found")
            else:
                print(f"   ❌ Missing job files: {missing_files}")
                
            # Check for models directory
            models_dir = os.path.join(job_path, 'models')
            if os.path.exists(models_dir):
                print("   ✅ Models directory found")
                # Check for actual trained models
                model_count = 0
                for arch_dir in os.listdir(models_dir):
                    arch_path = os.path.join(models_dir, arch_dir)
                    if os.path.isdir(arch_path):
                        for task_dir in os.listdir(arch_path):
                            task_path = os.path.join(arch_path, task_dir)
                            if os.path.exists(os.path.join(task_path, 'best_model.pth')):
                                model_count += 1
                print(f"   ✅ Found {model_count} trained models")
            else:
                print("   ❌ Models directory not found")
        
        # Test 4: Test GUI creation (without showing)
        print("4. Testing GUI creation...")
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication([])
            
            test_gui = TestSelectionGUI()
            print("   ✅ Test selection GUI created successfully")
            
            if job_folders and job_path:
                standalone_gui = VEstimStandaloneTestingGUI(job_path)
                print("   ✅ Standalone testing GUI created successfully")
            
            app.quit()
        except Exception as e:
            print(f"   ❌ GUI creation failed: {e}")
        
        print("\n🎯 Workflow validation complete!")
        print("To run the standalone testing:")
        print(f"   python {os.path.join(project_root, 'launch_standalone_testing_gui.py')}")
        print("\nWorkflow summary:")
        print("1. Select job folder (with trained models)")
        print("2. Select test data file (any CSV)")
        print("3. Apply augmentation if required (filters, etc.)")
        print("4. Run testing on all models in job")
        print("5. View results with plot functionality")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_standalone_testing()