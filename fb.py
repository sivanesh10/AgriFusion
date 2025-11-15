# list_models_v2.py
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY present:", bool(api_key))

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    print("Error constructing genai.Client:", e)
    raise SystemExit(1)

def try_list():
    # Try client.list_models if present
    if hasattr(client, "list_models"):
        try:
            models = client.list_models()
            print("client.list_models() returned:")
            for m in models:
                # m may be a dict-like or object
                name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None) or str(m)
                print(" -", name)
            return True
        except Exception as e:
            print("client.list_models() failed:", e)

    # Try client.models.list()
    try:
        resp = client.models.list()
        print("client.models.list() returned pager/response. Iterating...")
        # resp may be a pager with attribute .models OR an iterable
        # Try .models first
        if hasattr(resp, "models") and resp.models:
            for m in resp.models:
                name = getattr(m, "name", None) or str(m)
                print(" -", name)
            return True
        # Otherwise, iterate the pager
        try:
            for m in resp:
                name = getattr(m, "name", None) or str(m)
                print(" -", name)
            return True
        except Exception as e:
            print("Iterating resp failed:", e)
    except Exception as e:
        print("client.models.list() failed:", e)
    return False

ok = try_list()
if not ok:
    print("No models printed. If this happens, check billing / permissions in Google AI Studio.")
