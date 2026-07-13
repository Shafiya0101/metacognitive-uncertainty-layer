from pyngrok import ngrok
import subprocess
import sys
import time

# Start ngrok tunnel
public_url = ngrok.connect(7860)
print("\n" + "=" * 60)
print("  YOUR PUBLIC APP LINK:")
print(f"  {public_url}")
print("=" * 60)
print("\nShare this link with anyone!")
print("Keep this window open to keep the link alive.")
print("Press Ctrl+C to stop.\n")

# Run the app
subprocess.run([sys.executable, "app.py"])