"""
BTC Linear Regression Strategy - Live Dashboard
================================================
Flask web app for monitoring entry signals on mobile.
"""

import ccxt
import pandas as pd
import numpy as np
from scipy import stats
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# =============================================================================
# STRATEGY PARAMETERS (same as backtest)
# =============================================================================
LOOKBACK_PERIOD = 100
R2_THRESHOLD = 0.6
ENTRY_THRESHOLD_PCT = 0.5
STOP_MULTIPLIER = 2.0
TIMEFRAME = '5m'

# =============================================================================
# GLOBAL STATE
# =============================================================================
current_state = {
    'last_update': None,
    'btc_price': None,
    'cumulative_return': None,
    'regression_line': None,
    'channel_top': None,
    'channel_bottom': None,
    'entry_threshold': None,
    'r_squared': None,
    'signal': 'WAITING',
    'entry_price': None,
    'stop_loss': None,
    'take_profit': None,
    'error': None,
    'channel_top_price': None,
    'channel_bottom_price': None,
    'regression_line_price': None,
    'entry_threshold_price': None,
    'current_position_price': None,
}

# Store historical data
price_history = []
return_history = []

# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_btc_data():
    """Fetch latest BTC 5-minute candles from Kraken (works in US)."""
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})

        # Fetch last 150 candles (need 100 for lookback + buffer)
        ohlcv = exchange.fetch_ohlcv('BTC/USD', TIMEFRAME, limit=150)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_signals(df):
    """Calculate regression channel and check for entry signals."""
    global current_state

    if df is None or len(df) < LOOKBACK_PERIOD + 1:
        current_state['error'] = "Insufficient data"
        return

    try:
        # Calculate returns
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['returns'] = df['close'].pct_change()
        df = df.dropna()
        df['cumulative_return'] = df['returns'].cumsum()

        # Get the lookback window
        window = df['cumulative_return'].iloc[-LOOKBACK_PERIOD:].values
        x = np.arange(len(window))

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window)

        # Calculate regression line at current point
        regression_value = slope * LOOKBACK_PERIOD + intercept

        # Calculate residuals for channel
        fitted = slope * x + intercept
        residuals = window - fitted
        max_residual = residuals.max()
        min_residual = residuals.min()

        # Channel bounds
        channel_top = regression_value + max_residual
        channel_bottom = regression_value + min_residual

        # Entry threshold (50% toward bottom)
        entry_threshold = regression_value - ENTRY_THRESHOLD_PCT * (regression_value - channel_bottom)

        # Current values
        current_price = df['close'].iloc[-1]
        current_cumret = df['cumulative_return'].iloc[-1]
        r_squared = r_value ** 2

        # Calculate base price for converting returns to prices
        base_price = current_price / (1 + current_cumret) if current_cumret != -1 else current_price

        # Update state
        current_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_state['btc_price'] = float(current_price)
        current_state['cumulative_return'] = float(current_cumret)
        current_state['regression_line'] = float(regression_value)
        current_state['channel_top'] = float(channel_top)
        current_state['channel_bottom'] = float(channel_bottom)
        current_state['entry_threshold'] = float(entry_threshold)
        current_state['r_squared'] = float(r_squared)
        current_state['error'] = None

        # Add price levels for display
        current_state['channel_top_price'] = float(base_price * (1 + channel_top))
        current_state['channel_bottom_price'] = float(base_price * (1 + channel_bottom))
        current_state['regression_line_price'] = float(base_price * (1 + regression_value))
        current_state['entry_threshold_price'] = float(base_price * (1 + entry_threshold))
        current_state['current_position_price'] = float(current_price)

        # Check entry conditions
        if r_squared > R2_THRESHOLD and current_cumret <= entry_threshold:
            current_state['signal'] = 'ENTRY SIGNAL'

            # Calculate SL and TP
            stop_distance = current_cumret - channel_bottom
            stop_loss_cumret = current_cumret - (stop_distance * STOP_MULTIPLIER)
            take_profit_cumret = regression_value

            # Convert to approximate price levels
            # This is approximate - for display purposes
            price_per_return = current_price / (1 + current_cumret) if current_cumret != 0 else current_price

            current_state['entry_price'] = float(current_price)
            current_state['stop_loss'] = float(current_price * (1 + stop_loss_cumret - current_cumret))
            current_state['take_profit'] = float(current_price * (1 + take_profit_cumret - current_cumret))
        else:
            current_state['signal'] = 'WAITING'
            current_state['entry_price'] = None
            current_state['stop_loss'] = None
            current_state['take_profit'] = None

    except Exception as e:
        current_state['error'] = str(e)
        print(f"Error calculating signals: {e}")

def update_loop():
    """Background thread to update data every 60 seconds."""
    while True:
        try:
            df = fetch_btc_data()
            calculate_signals(df)
            r2_display = f"{current_state['r_squared']:.3f}" if current_state['r_squared'] else 'N/A'
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated - Signal: {current_state['signal']}, R²: {r2_display}")
        except Exception as e:
            print(f"Update error: {e}")
        time.sleep(60)  # Update every 60 seconds

# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify(current_state)

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/refresh', methods=['POST'])
def refresh():
    """Manually trigger data refresh."""
    try:
        df = fetch_btc_data()
        calculate_signals(df)
        return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# =============================================================================
# STARTUP - runs when gunicorn loads the app
# =============================================================================
# Start background update thread
update_thread = threading.Thread(target=update_loop, daemon=True)
update_thread.start()

# Initial update
try:
    df = fetch_btc_data()
    calculate_signals(df)
except Exception as e:
    print(f"Initial fetch error: {e}")

# =============================================================================
# MAIN - for local development only
# =============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
