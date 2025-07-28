# core/order_simulator.py

def simulate_trade(signal, capital, leverage, slippage_rate, fee_rate, entry_price):
    """
    슬리피지 및 수수료를 반영한 진입 시뮬레이션

    Parameters:
    - signal: dict, {"direction": "LONG" or "SHORT"}
    - capital: 진입 자본
    - leverage: 레버리지
    - slippage_rate: 슬리피지 비율 (예: 0.001)
    - fee_rate: 거래 수수료율 (예: 0.0005)
    - entry_price: 현재 시세

    Returns:
    - dict: {
        "entry_price": 체결가,
        "direction": "LONG"/"SHORT",
        "capital": 사용 자본,
        "leverage": 사용 레버리지,
        "fee": 진입 수수료,
        "position_size": 보유 수량,
    }
    """
    direction = signal.get("direction", "LONG")
    slippage = entry_price * slippage_rate
    filled_price = entry_price + slippage if direction == "LONG" else entry_price - slippage

    fee = capital * fee_rate * leverage
    position_size = (capital * leverage) / filled_price

    return {
        "entry_price": filled_price,
        "direction": direction,
        "capital": capital,
        "leverage": leverage,
        "fee": fee,
        "position_size": position_size
    }

def simulate_exit(position, exit_price, slippage_rate=0.001, fee_rate=0.0005):
    """
    포지션 청산 시 수익 계산 (슬리피지 및 수수료 반영)

    Parameters:
    - position: dict, simulate_trade의 리턴값
    - exit_price: 청산 시 시세
    - slippage_rate: 슬리피지
    - fee_rate: 수수료

    Returns:
    - dict: {
        "exit_price": 실청산가,
        "pnl": 손익,
        "roi": 수익률,
        "fee": 청산 수수료,
        "net_profit": 순익
    }
    """
    direction = position["direction"]
    position_size = position["position_size"]
    entry_price = position["entry_price"]
    capital = position["capital"]
    leverage = position["leverage"]

    # 슬리피지 반영 청산가
    slippage = exit_price * slippage_rate
    filled_exit = exit_price - slippage if direction == "LONG" else exit_price + slippage

    # 수익 계산
    if direction == "LONG":
        gross_profit = (filled_exit - entry_price) * position_size
    else:
        gross_profit = (entry_price - filled_exit) * position_size

    fee_exit = capital * fee_rate * leverage
    total_fee = position["fee"] + fee_exit
    net_profit = gross_profit - total_fee
    roi = net_profit / capital

    return {
        "exit_price": filled_exit,
        "pnl": gross_profit,
        "roi": roi,
        "fee": total_fee,
        "net_profit": net_profit
    }
