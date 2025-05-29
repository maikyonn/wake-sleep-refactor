#!/usr/bin/env python
"""
Test script to verify the new repository structure works correctly.
"""

import sys
import os

def test_imports():
    """Test that key imports work in the new structure."""
    print("Testing imports...")
    
    try:
        # Test tokenizer import
        from staria.models.tokenizer import MusicTokenizerWithStyle
        print("✅ Tokenizer import successful")
        
        # Test tokenizer initialization
        tokenizer = MusicTokenizerWithStyle()
        print(f"✅ Tokenizer initialized - vocab_size: {tokenizer.vocab_size}")
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False
    
    try:
        # Test decoder baseline import
        from staria.baselines.decoder_only import DecoderOnlyBaseline
        print("✅ DecoderOnlyBaseline import successful")
        
        # Test baseline initialization
        model = DecoderOnlyBaseline(vocab_size=tokenizer.vocab_size)
        print(f"✅ DecoderOnlyBaseline initialized - parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Decoder baseline test failed: {e}")
        return False
    
    print("\n🎉 All import tests passed!")
    return True

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "staria",
        "staria/models",
        "staria/baselines", 
        "staria/data",
        "staria/generation",
        "staria/evaluation",
        "staria/utils",
        "scripts",
        "external",
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            return False
    
    # Test key files exist
    key_files = [
        "staria/models/tokenizer.py",
        "staria/baselines/decoder_only.py",
        "scripts/train_decoder_only.py",
        "scripts/generate_decoder_only.py",
        "external/ariautils/config/config.json"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    print("\n🎉 Directory structure test passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("STARIA REPOSITORY STRUCTURE TEST")
    print("=" * 50)
    
    structure_ok = test_directory_structure()
    imports_ok = test_imports()
    
    if structure_ok and imports_ok:
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Repository structure is working correctly.")
        print("=" * 50)
        return True
    else:
        print("\n" + "=" * 50)
        print("❌ TESTS FAILED! Please check the issues above.")
        print("=" * 50)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)