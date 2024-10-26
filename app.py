from flask import Flask, render_template, request
from lstm_model import fetch_stock_data, train_lstm, predict_and_mark
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    chart = None
    if request.method == 'POST':
        ticker = request.form['ticker']
        data = fetch_stock_data(ticker)
        model, scaler = train_lstm(data)
        markers = predict_and_mark(data, model, scaler)
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close']
        )])
        for marker in markers:
            signal, date, price = marker
            color = 'green' if signal == 'buy' else 'red'
            arrow_y = price * 1.02 if signal == 'buy' else price * 0.98  # Position arrows slightly above/below
            fig.add_annotation(
                x=date, y=arrow_y,
                xref='x', yref='y',
                text='▲' if signal == 'buy' else '▼',
                showarrow=False,
                font=dict(color=color, size=16)
            )
        chart = fig.to_html(full_html=False)
    return render_template('index.html', chart=chart)

if __name__ == '__main__':
    app.run(debug=True)