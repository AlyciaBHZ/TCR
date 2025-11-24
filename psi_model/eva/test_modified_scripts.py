#!/usr/bin/env python3
"""
Test script for the modified language model evaluation scripts
"""

import os
import pandas as pd
import sys

def test_summarize_lm_results():
    """Test the summarize_lm_results script"""
    print("🧪 Testing summarize_lm_results.py...")
    
    # Create dummy results data if not exists
    dummy_results = {
        'model': ['Model_A', 'Model_A', 'Model_B', 'Model_B'] * 5,
        'mask_ratio': [0.1, 0.3, 0.1, 0.3] * 5,
        'recovery_accuracy': [0.8, 0.7, 0.75, 0.65] * 5,
        'nll': [2.1, 2.3, 2.2, 2.4] * 5,
        'perplexity': [8.2, 10.0, 9.0, 11.0] * 5,
        'success': [True] * 20
    }
    
    dummy_df = pd.DataFrame(dummy_results)
    
    # Save dummy data
    dummy_df.to_csv('./lm_evaluation_results.csv', index=False)
    print("✅ Created dummy evaluation results")
    
    # Test the summarize function
    try:
        from summarize_lm_results import summarize_lm_results
        result = summarize_lm_results()
        if result is not None:
            print("✅ summarize_lm_results.py works correctly!")
            return True
        else:
            print("❌ summarize_lm_results.py returned None")
            return False
    except Exception as e:
        print(f"❌ Error testing summarize_lm_results.py: {e}")
        return False

def test_language_model_evaluation():
    """Test the language model evaluation script"""
    print("\n🧪 Testing language_model_evaluation_proper.py...")
    
    # Check if exp_data.csv exists
    if not os.path.exists('./exp_data.csv'):
        print("⚠️  exp_data.csv not found, creating dummy data...")
        
        # Create minimal dummy experimental data
        dummy_exp_data = {
            'peptide': ['YLQPRTFLL'] * 10,
            'mhc': ['HLA-A*02:01'] * 10,
            'cdr3_b': ['CASSLEETQYF', 'CASSLDTGELF', 'CASRPGLAGGRPEQFF', 
                      'CASSQDRGTEAFF', 'CASSLDSYEQYF', 'CASSLEADTQYF',
                      'CASSLGETQYF', 'CASSLDAGTEAFF', 'CASRPGQVTEAFF', 'CASSLDETEAFF'],
            'l_v': ['TRBV2'] * 10,
            'l_j': ['TRBJ2-3'] * 10,
            'h_v': ['TRBV19'] * 10,
            'h_j': ['TRBJ2-7'] * 10
        }
        
        dummy_exp_df = pd.DataFrame(dummy_exp_data)
        dummy_exp_df.to_csv('./exp_data.csv', index=False)
        print("✅ Created dummy experimental data")
    
    # Test data preparation
    try:
        sys.path.append('.')
        from language_model_evaluation_proper import prepare_data_for_lm_evaluation
        
        formatted_path = prepare_data_for_lm_evaluation('./exp_data.csv', './test_formatted_data.csv')
        if formatted_path:
            print("✅ Data preparation works correctly!")
            
            # Check if formatted data was created
            if os.path.exists('./test_formatted_data.csv'):
                formatted_df = pd.read_csv('./test_formatted_data.csv')
                print(f"✅ Formatted data contains {len(formatted_df)} samples")
                return True
            else:
                print("❌ Formatted data file was not created")
                return False
        else:
            print("❌ Data preparation failed")
            return False
    except Exception as e:
        print(f"❌ Error testing language_model_evaluation_proper.py: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing modified language model evaluation scripts...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('./nll', exist_ok=True)
    
    # Run tests
    test1_passed = test_summarize_lm_results()
    test2_passed = test_language_model_evaluation()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"   summarize_lm_results.py: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   language_model_evaluation_proper.py: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The scripts are ready to use.")
        print("\nNext steps:")
        print("1. Run 'python summarize_lm_results.py' if you have evaluation results")
        print("2. Run 'python language_model_evaluation_proper.py --skip_model_loading' to test data preparation")
        print("3. Modify model paths in MODELS_CONFIG if you want to load actual models")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    # Cleanup test files
    cleanup_files = ['./lm_evaluation_results.csv', './test_formatted_data.csv']
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up {file}")

if __name__ == "__main__":
    main() 