
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load assets
log_model = joblib.load("aci_logistic_model.pkl")
tree_model = joblib.load("aci_decision_tree_model.pkl")
scaler = joblib.load("aci_scaler.pkl")

features_to_scale = [
    "buy_price",
    "sell_price",
    "fee_percent",
    "slippage_percent",
    "raw_spread",
    "adj_spread",
    "estimated_profit"
]

app = FastAPI()

class TradeInput(BaseModel):
    buy_price: float
    sell_price: float
    fee_percent: float
    slippage_percent: float

@app.post("/predict_trade")
def predict_trade(trade: TradeInput):
    raw_spread = ((trade.sell_price - trade.buy_price) / trade.buy_price) * 100
    adj_spread = raw_spread - (trade.fee_percent + trade.slippage_percent)
    estimated_profit = max(0, adj_spread / 100 * 1000)

    df = pd.DataFrame([{
        "buy_price": trade.buy_price,
        "sell_price": trade.sell_price,
        "fee_percent": trade.fee_percent,
        "slippage_percent": trade.slippage_percent,
        "raw_spread": raw_spread,
        "adj_spread": adj_spread,
        "estimated_profit": estimated_profit
    }])
    df[features_to_scale] = scaler.transform(df[features_to_scale])

    log_pred = int(log_model.predict(df)[0])
    tree_pred = int(tree_model.predict(df)[0])
    decision = "EXECUTE" if log_pred == 1 and tree_pred == 1 else "SKIP"

    return {
        "logistic_prediction": log_pred,
        "tree_prediction": tree_pred,
        "agreement": log_pred == tree_pred,
        "final_decision": decision
    }
