#!/usr/bin/env python3
"""
Quick test to validate boundary format parsing
"""

def test_boundary_validation():
    """Test the boundary format validation logic"""
    
    def _validate_boundary_format(value, param_name):
        """Test version of boundary validation"""
        if not value or value.strip() == "":
            return False, f"{param_name} cannot be empty for Optuna optimization.\nCurrent value: '{value}'"
        
        value = value.strip()
        
        # Check for boundary format [min,max]
        if value.startswith('[') and value.endswith(']'):
            try:
                # Remove brackets and split by comma
                inner = value[1:-1].strip()
                parts = [part.strip() for part in inner.split(',')]
                
                if len(parts) != 2:
                    return False, f"{param_name} must have exactly 2 values in format [min,max].\nCurrent value: '{value}'"
                
                # Try to convert to numbers
                min_val = float(parts[0])
                max_val = float(parts[1])
                
                if min_val >= max_val:
                    return False, f"{param_name}: min value must be less than max value.\nCurrent value: '{value}'"
                
                return True, f"Valid boundary format for {param_name}"
                
            except ValueError:
                return False, f"{param_name} must contain numeric values in format [min,max].\nCurrent value: '{value}'"
        else:
            return False, f"{param_name} must be in boundary format [min,max] for Optuna optimization.\nCurrent value: '{value}'"

    # Test cases
    test_cases = [
        ("INITIAL_LR", "[0.001,0.1]", True),
        ("INITIAL_LR", "[0.001, 0.1]", True),  # With spaces
        ("INITIAL_LR", "[ 0.001 , 0.1 ]", True),  # More spaces
        ("INITIAL_LR", "0.001", False),  # Single value
        ("INITIAL_LR", "[0.001]", False),  # Only one value
        ("INITIAL_LR", "[0.1,0.001]", False),  # Min > Max
        ("HIDDEN_UNITS", "[10,100]", True),
        ("LAYERS", "[1,5]", True),
        ("MAX_EPOCHS", "[50,200]", True),
        ("BATCH_SIZE", "64", False),  # Single value should fail
    ]
    
    print("🧪 Testing Boundary Format Validation")
    print("=" * 50)
    
    for param_name, value, expected_valid in test_cases:
        is_valid, message = _validate_boundary_format(value, param_name)
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
        
        print(f"{status} {param_name}: '{value}' -> {is_valid}")
        if not is_valid == expected_valid:
            print(f"    Expected: {expected_valid}, Got: {is_valid}")
            print(f"    Message: {message}")
        print()

if __name__ == "__main__":
    test_boundary_validation()
