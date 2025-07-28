from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()
    return {
        "slippage_weight": float(os.getenv("PENALTY_SLIPPAGE_WEIGHT", 3.0)),
        "volatility_weight": float(os.getenv("PENALTY_VOLATILITY_WEIGHT", 10.0)),
    }