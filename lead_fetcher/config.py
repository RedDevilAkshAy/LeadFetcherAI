import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GOOGLE_URL = os.getenv("GOOGLE_URL", "https://places.googleapis.com/v1/places")
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY", "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")
NVIDIA_EMBEDDING_MODEL=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
NVIDIA_MODEL=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
NVIDIA_BASE_URL=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")