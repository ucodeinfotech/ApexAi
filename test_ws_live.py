"""Angel One WebSocket V2 Live Test"""
import pyotp, json, time, struct
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# ─── CREDENTIALS ───
API_KEY = "DPUfQ4dz"
CLIENT_ID = "D52359454"
PASSWORD = "1234"
TOTP_SECRET = "UALX2FAJSGYMFTKYHHYT4HC3IE"

# ─── LOGIN ───
print("Logging in...")
smart = SmartConnect(api_key=API_KEY)
session = smart.generateSession(CLIENT_ID, PASSWORD, pyotp.TOTP(TOTP_SECRET).now())
if not session.get("status"):
    print("Login failed:", session)
    exit(1)

auth_token = session["data"]["jwtToken"]
feed_token = session["data"]["feedToken"]
print(f"Logged in. Feed token: {feed_token[:10]}..., Auth token: {auth_token[:20]}...")

user = smart.getProfile(session["data"]["refreshToken"])
print(f"User: {user.get('data', {}).get('clientcode', '?')}")

# ─── WEBSOCKET ───
print("\nConnecting WebSocket V2...")
socket = SmartWebSocketV2(
    auth_token=auth_token,
    api_key=API_KEY,
    client_code=CLIENT_ID,
    feed_token=feed_token,
    max_retry_attempt=2,
)

received = []

def on_data(wsapp, data):
    token = data.get("token", "")
    ltp = data.get("last_traded_price", 0) / 100
    ts = data.get("exchange_timestamp", 0)
    tm = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "?"
    mode = data.get("subscription_mode_val", "?")
    print(f"[{tm}] {mode} token={token} LTP={ltp:.2f}")
    received.append((token, ltp))
    if len(received) >= 5:
        print("\nGot 5 ticks. Closing.")
        socket.close_connection()

def on_open(wsapp):
    print("WebSocket CONNECTED. Subscribing...")
    socket.subscribe(
        correlation_id="testws0001",
        mode=2,  # QUOTE
        token_list=[{"exchangeType": 1, "tokens": ["26000", "26009", "3045", "2885", "1333"]}]
    )
    print("Subscribed to NIFTY, BANKNIFTY, SBIN, RELIANCE, HDFCBANK")

def on_error(wsapp, error):
    print(f"Error: {error}")

def on_close(wsapp):
    print("WebSocket CLOSED")

socket.on_open = on_open
socket.on_data = on_data
socket.on_error = on_error
socket.on_close = on_close

# Run for 15 seconds then close
import threading
def timeout():
    time.sleep(15)
    if socket.wsapp:
        print("\nTimeout reached. Closing.")
        socket.close_connection()

threading.Thread(target=timeout, daemon=True).start()
socket.connect()
