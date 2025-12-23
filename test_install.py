print("Testing Python 3.13 installation...")

# Test basic imports
try:
    import streamlit as st
    print("✅ streamlit: OK")
except Exception as e:
    print(f"❌ streamlit: {e}")

try:
    import cohere
    print("✅ cohere: OK")
except Exception as e:
    print(f"❌ cohere: {e}")

try:
    import pypdf
    print("✅ pypdf: OK")
except Exception as e:
    print(f"❌ pypdf: {e}")

try:
    import numpy as np
    print(f"✅ numpy: OK (version {np.__version__})")
except Exception as e:
    print(f"❌ numpy: {e}")

try:
    import pandas as pd
    print(f"✅ pandas: OK (version {pd.__version__})")
except Exception as e:
    print(f"❌ pandas: {e}")

print("\n🎉 If you see all ✅, you're ready to proceed!")
