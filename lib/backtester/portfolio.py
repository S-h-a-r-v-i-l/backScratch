"""
Tracks positions, cash, and running P&L. Handles position sizing (e.g.,
vol-targeting based on HAR-RV forecasts), transaction costs, and mark-to-market.
Separate from engine.py so sizing rules can be swapped without rewriting the loop.
"""

class Portfolio:
    def __init__(self, cash, shares, equity, peak_equity, cumulative_pnl):
        self.cash = cash
        self.shares = shares
        self.equity = equity
        self.peak_equity = peak_equity
        self.cumulative_pnl = cumulative_pnl

    def calc_position_size(self, prev_weight: float, curr_weight: float) -> float:
        return (curr_weight - prev_weight) * self.equity

    def get_borrowed(self, price: float) -> float:
        position_value = self.shares * price
        return max(0., position_value - self.equity)

    def get_leverage(self, price: float) -> float:
        position_value = self.shares * price
        return position_value / self.equity if self.equity > 0 else 0.

    def execute_trade(self, trade_size: float, price: float, transaction_cost: float) -> None:
        if trade_size == 0:
            return

        prev_equity = self.equity
        self.cash -= trade_size + transaction_cost
        self.shares += trade_size / price
        self.equity = self.cash + self.shares * price
        self.peak_equity = max(self.peak_equity, self.equity)
        self.cumulative_pnl += self.equity - prev_equity

    def accrue_interest(self, price: float, borrow_rate: float, cash_rate: float) -> None:
        prev_equity = self.equity
        borrowed = self.get_borrowed(price)

        if borrowed > 0:
            interest = -borrowed * (borrow_rate / 365)
        else:
            interest = self.cash * (cash_rate / 365)

        self.cash += interest
        self.equity = self.cash + self.shares * price
        self.peak_equity = max(self.peak_equity, self.equity)
        self.cumulative_pnl += self.equity - prev_equity

    def update_equity(self, price: float) -> None:
        prev_equity = self.equity
        self.equity = self.cash + self.shares * price
        self.peak_equity = max(self.peak_equity, self.equity)
        self.cumulative_pnl += self.equity - prev_equity

    def calc_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def update(self, price: float, borrow_rate: float, cash_rate: float) -> None:
        self.accrue_interest(price, borrow_rate, cash_rate)
        self.update_equity(price)

    def get_state(self) -> dict:
        return {
            "cash": self.cash,
            "shares": self.shares,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "cumulative_pnl": self.cumulative_pnl,
            "drawdown": self.calc_drawdown()
        }

    


    


