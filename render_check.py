import os
import sys
import platform
import importlib

def check_render_compatibility():
    """Check for common issues that might cause problems on Render"""
    issues = []
    
    # Check platform-specific code
    if platform.system() == "Windows":
        print("⚠️ Running on Windows - Render uses Linux")
        
    # Check for Linux-only modules
    try:
        import resource
        print("✅ Resource module available")
    except ImportError:
        issues.append("❌ Resource module not available on Windows but used on Linux")
    
    # Check for signal handling
    import signal
    if not hasattr(signal, 'SIGALRM'):
        issues.append("❌ SIGALRM not available on Windows but might be used in your code")
    
    # Check for ctypes and libc
    if platform.system() == "Windows":
        issues.append("⚠️ Linux-specific memory management with libc.so.6 won't work on Windows")
    
    # Check that all imports will work
    modules_to_check = [
        "document_processor", 
        "retrieval_engine_extended", 
        "prompt_engine",
        "enhanced_capabilities.capability_router",
        "enhanced_capabilities.conversation_memory"
    ]
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ Module {module} importable")
        except ImportError as e:
            issues.append(f"❌ Module {module} import error: {str(e)}")
    
    return issues

if __name__ == "__main__":
    print("=== Checking Render Compatibility ===")
    issues = check_render_compatibility()
    
    if issues:
        print("\n=== Potential Render Deployment Issues ===")
        for issue in issues:
            print(issue)
        print("\nFix these issues before deploying to Render")
    else:
        print("\n✅ No obvious compatibility issues detected")