from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def buy_sell(data_frame, portfolio_value, commission, stop_loss):
    buying_power = portfolio_value
    if stop_loss == 0:
        stop_loss_percent = 0
    else:
        stop_loss_percent = 1 - (stop_loss / 100)
    buy_price = 0
    shares = 0
    value = 0
    for i in range(0, len(data_frame['Buy/Sell'])):
        date = data_frame[
            (data_frame['Close'] == data_frame.ix[i]['Close']) & (data_frame['Open'] == data_frame.ix[i]['Open'])].index
        data_frame.set_value(date, 'Own', value)
        data_frame.set_value(date, 'Shares', shares)
        data_frame.set_value(date, 'Buying Power', buying_power)
        if data_frame.ix[i]['Buy/Sell'] == 1 and data_frame.ix[i]['Own'] == 0:
            value = 1
            buying_power -= commission
            shares = math.floor(buying_power / data_frame.ix[i]['Open'])
            buying_power = buying_power - shares * data_frame.ix[i]['Open']
            data_frame.set_value(date, 'Buy or Sell', 1)
            buy_price = data_frame.ix[i]['Open']
        elif data_frame.ix[i]['Buy/Sell'] == -1 and data_frame.ix[i]['Own'] == 1:
            value = 0
            buying_power -= commission
            buying_power = buying_power + shares * data_frame.ix[i]['Open']
            shares = 0
            data_frame.set_value(date, 'Buy or Sell', -1)
        # Stoploss condition
        if data_frame.ix[i]['Close'] <= buy_price * stop_loss_percent and data_frame.ix[i]['Own'] == 1:
            value = 0
            buying_power -= commission
            buying_power = buying_power + shares * data_frame.ix[i]['Open']
            shares = 0
            data_frame.set_value(date, 'Buy or Sell', -1)
    data_frame.drop('Buy/Sell', axis=1, inplace=True)
    data_frame['Portfolio'] = data_frame['Buying Power'] + data_frame['Shares'] * data_frame['Open']


def sma_fast_slow():
    while True:
        fast = input("Please enter fast moving average day count if left blank will be set to 5 days: ")
        if not fast:
            fast = 5
            break
        elif fast.isnumeric():
            fast = int(fast)
            break
        else:
            print('Fast moving average was entered incorrectly')
    while True:
        slow = input("Please enter slow moving average day count if left blank will be set to 20 days: ")
        if not slow:
            slow = 20
            break
        elif slow.isnumeric():
            slow = int(slow)
            break
        else:
            print('Slow moving average was entered incorrectly')
    return fast, slow


def sma(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    fast, slow = sma_fast_slow()
    stock__s_m_a = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power'])
    stock__s_m_a = stock__s_m_a.join(new_col)

    stock__s_m_a['Fast SMA'] = stock__s_m_a['Close'].rolling(window=fast).mean()
    stock__s_m_a['Slow SMA'] = stock__s_m_a['Close'].rolling(window=slow).mean()

    fast_v_slow = stock__s_m_a['Fast SMA'] - stock__s_m_a['Slow SMA']

    # Positive when FSMA is above SSMA, negative when FSMA is below SSMA
    inc_dec__s_m_a = np.sign(fast_v_slow)
    stock__s_m_a['Buy/Sell'] = np.sign(inc_dec__s_m_a.diff())

    buy_sell(stock__s_m_a, portfolio_initial_value, commission, stop_loss_percentage)

    buy__s_m_a = stock__s_m_a[stock__s_m_a['Buy or Sell'] == 1]
    sell__s_m_a = stock__s_m_a[stock__s_m_a['Buy or Sell'] == -1]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    stock__s_m_a['Close'].plot(label='{} Close'.format(stock_name), color='k')
    plt.title('{} Closing Price and {} Day Moving Average'.format(stock_name, fast))
    plt.ylabel('Share Price')
    if buy__s_m_a.empty == 0:
        plt.plot(buy__s_m_a.index, buy__s_m_a['Close'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__s_m_a.empty == 0:
        plt.plot(sell__s_m_a.index, sell__s_m_a['Close'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.legend(loc='upper left')

    plt.subplot(312)
    stock__s_m_a['Fast SMA'].plot(label='{} Day Avg'.format(fast), color='c')
    stock__s_m_a['Slow SMA'].plot(label='{} Day Avg'.format(slow), color='m')
    plt.title('{} - {} Day Moving Average and {} Day Moving Average'.format(stock_name, fast, slow))
    if buy__s_m_a.empty == 0:
        plt.plot(buy__s_m_a.index, buy__s_m_a['Fast SMA'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__s_m_a.empty == 0:
        plt.plot(sell__s_m_a.index, sell__s_m_a['Fast SMA'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.legend(loc='upper left')

    plt.subplot(313)
    plt.plot(stock__s_m_a['Portfolio'], label='Portfolio Value', color='k')
    plt.title('Portfolio Value')
    if buy__s_m_a.empty == 0:
        plt.plot(buy__s_m_a.index, buy__s_m_a['Portfolio'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__s_m_a.empty == 0:
        plt.plot(sell__s_m_a.index, sell__s_m_a['Portfolio'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.title('Portfolio Value'.format(stock_name, fast, slow))
    plt.xlabel('Date')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def macd(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    EMA = input("If you would like to set custom MACD values enter as [#,#,#], otherwise leave blank: ")
    e_m_a_len = [12, 26, 9]

    stock__m_a_c_d = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power'])
    stock__m_a_c_d = stock__m_a_c_d.join(new_col)

    e_m_a_12 = 2 / (e_m_a_len[0] + 1) * (stock__m_a_c_d['Close'][e_m_a_len[0]]) + (1 - 2 / (e_m_a_len[0] + 1)) * \
                                                                                  stock__m_a_c_d[
                                                                                      'Close'][
                                                                                  0:e_m_a_len[
                                                                                      0]].mean()
    e_m_a_26 = 2 / (e_m_a_len[1] + 1) * (stock__m_a_c_d['Close'][e_m_a_len[1]]) + (1 - 2 / (e_m_a_len[1] + 1)) * \
                                                                                  stock__m_a_c_d[
                                                                                      'Close'][
                                                                                  0:e_m_a_len[
                                                                                      1]].mean()

    ema12 = []
    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i < e_m_a_len[0]:
            ema12.append(np.nan)
        else:
            ema12.append(e_m_a_12)
            e_m_a_12 = (2 / (e_m_a_len[0] + 1)) * stock__m_a_c_d['Close'][i] + (1 - 2 / (e_m_a_len[0] + 1)) * e_m_a_12

    ema26 = []
    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i <= e_m_a_len[1]:
            ema26.append(np.nan)
        else:
            ema26.append(e_m_a_26)
            e_m_a_26 = (2 / (e_m_a_len[1] + 1)) * stock__m_a_c_d['Close'][i] + (1 - 2 / (e_m_a_len[1] + 1)) * e_m_a_26

    stock__m_a_c_d['MACD'] = np.array(ema12) - np.array(ema26)
    signal_val = 2 / (e_m_a_len[2] + 1) * (stock__m_a_c_d['MACD'][e_m_a_len[1] + e_m_a_len[2]]) + (1 - 2 / (
        e_m_a_len[2] + 1)) * \
                                                                                                  stock__m_a_c_d[
                                                                                                      'MACD'][
                                                                                                  e_m_a_len[1]:
                                                                                                  e_m_a_len[
                                                                                                      1] +
                                                                                                  e_m_a_len[
                                                                                                      2]].mean()
    signal = []
    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i <= e_m_a_len[1] + e_m_a_len[2]:
            signal.append(np.nan)
        else:
            signal.append(signal_val)
            signal_val = (2 / (e_m_a_len[2] + 1)) * stock__m_a_c_d['MACD'][i] + (
                                                                                    1 - 2 / (
                                                                                        e_m_a_len[2] + 1)) * signal_val

    stock__m_a_c_d['Signal'] = signal
    stock__m_a_c_d['MACD_Hist'] = (stock__m_a_c_d['MACD'] - stock__m_a_c_d['Signal'])

    inc_dec_MACD = np.sign(stock__m_a_c_d['MACD_Hist'])
    stock__m_a_c_d['Buy/Sell'] = np.sign(inc_dec_MACD.diff())

    buy_sell(stock__m_a_c_d, portfolio_initial_value, commission, stop_loss_percentage)
    buy__m_a_c_d = stock__m_a_c_d[stock__m_a_c_d['Buy or Sell'] == 1]
    sell__m_a_c_d = stock__m_a_c_d[stock__m_a_c_d['Buy or Sell'] == -1]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    stock__m_a_c_d['Close'].plot(label='{} Close'.format(stock_name), color='k')
    plt.title('{} Closing Price'.format(stock_name))
    plt.ylabel('Share Price')
    if buy__m_a_c_d.empty == 0:
        plt.plot(buy__m_a_c_d.index, buy__m_a_c_d['Close'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__m_a_c_d.empty == 0:
        plt.plot(sell__m_a_c_d.index, sell__m_a_c_d['Close'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.legend(loc='upper left')

    plt.subplot(312)
    stock__m_a_c_d['MACD'].plot(label='MACD', color='c')
    stock__m_a_c_d['Signal'].plot(label='Signal', color='m')
    plt.title('{} - MACD and Signal'.format(stock_name))
    if buy__m_a_c_d.empty == 0:
        plt.plot(buy__m_a_c_d.index, buy__m_a_c_d['Signal'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__m_a_c_d.empty == 0:
        plt.plot(sell__m_a_c_d.index, sell__m_a_c_d['Signal'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.legend(loc='upper left')

    plt.subplot(313)
    plt.plot(stock__m_a_c_d['Portfolio'], label='Portfolio Value', color='k')
    plt.title('Portfolio Value')
    if buy__m_a_c_d.empty == 0:
        plt.plot(buy__m_a_c_d.index, buy__m_a_c_d['Portfolio'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__m_a_c_d.empty == 0:
        plt.plot(sell__m_a_c_d.index, sell__m_a_c_d['Portfolio'], marker='v', markersize=10, linestyle="None",
                 color='r',
                 label='Sell')
    plt.xlabel('Date')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def bbands(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    stock__b_bands = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power'])
    stock__b_bands = stock__b_bands.join(new_col)

    h_l_c = (stock__b_bands['High'] + stock__b_bands['Low'] + stock__b_bands['Close']) / 3

    twenty_day_avg = h_l_c.rolling(window=20).mean()
    twenty_day_std = h_l_c.rolling(window=20).std()

    # Calculating Upper and Lower Band
    upp_band = twenty_day_avg + 2 * twenty_day_std
    low_band = twenty_day_avg - 2 * twenty_day_std

    # Buy and Sell indicators, based on when above or below Bollinger Bands
    over = ((stock__b_bands['Close'] - upp_band).apply(lambda x: x > 0))
    under = ((stock__b_bands['Close'] - low_band).apply(lambda x: x < 0))

    stock__b_bands['Buy/Sell'] = under + (-1) * over
    buy_sell(stock__b_bands, portfolio_initial_value, commission, stop_loss_percentage)

    buy__b_bands = stock__b_bands[stock__b_bands['Buy or Sell'] == 1]
    sell__b_bands = stock__b_bands[stock__b_bands['Buy or Sell'] == -1]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    stock__b_bands['Close'].plot(label='{} Close'.format(stock_name), color='k')
    upp_band.plot(label='Upper Band', color='b', linestyle='--')
    low_band.plot(label='Lower Band', color='b', linestyle='--')
    if buy__b_bands.empty == 0:
        plt.plot(buy__b_bands.index, buy__b_bands['Close'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__b_bands.empty == 0:
        plt.plot(sell__b_bands.index, sell__b_bands['Close'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.title('{} - Closing Price with Bollinger Bands'.format(stock_name))
    plt.ylabel('Share Price')
    plt.legend(loc='upper left')

    plt.subplot(312)
    twenty_day_avg.plot(label='Middle Band', color='Grey', linestyle='--')
    upp_band.plot(label='Upper Band', color='b', linestyle='--')
    low_band.plot(label='Lower Band', color='b', linestyle='--')
    plt.title('{} - Bollinger Bands'.format(stock_name))
    plt.legend(loc='upper left')

    plt.subplot(313)
    plt.plot(stock__b_bands['Portfolio'], label='Portfolio Value', color='k')
    plt.title('Portfolio Value')
    if buy__b_bands.empty == 0:
        plt.plot(buy__b_bands.index, buy__b_bands['Portfolio'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell__b_bands.empty == 0:
        plt.plot(sell__b_bands.index, sell__b_bands['Portfolio'], marker='v', markersize=10, linestyle="None",
                 color='r',
                 label='Sell')
    plt.xlabel('Date')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def rsi(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    Top_BBand = 70
    Lower_BBand = 30
    interval = 14

    Stock_RSI = stock.copy()
    New_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power'])
    Stock_RSI = Stock_RSI.join(New_col)

    # Determing gains and losses on daily basis. Binning gains and losses
    diff_close = Stock_RSI['Close'].diff()
    rs_pos = diff_close.apply(lambda x: x > 0) * diff_close
    rs_neg = diff_close.apply(lambda x: x < 0) * diff_close * -1

    rs_positive = rs_pos[0:interval].sum() / interval
    rs_negative = rs_neg[0:interval].sum() / interval

    # Smoothed Average Gain and Average Loss calculation
    rsp = []
    rsn = []
    for i in range(0, len(Stock_RSI['Close'])):
        if i > 13:
            rsp.append(rs_positive)
            rsn.append(rs_negative)
            rs_positive = (rs_positive * 13 + rs_pos[i]) / interval
            rs_negative = (rs_negative * 13 + rs_neg[i]) / interval
        else:
            rsp.append(0)
            rsn.append(0)

    RS = np.array(rsp) / np.array(rsn)
    Stock_RSI['RSI'] = 100 - (100 / (1 + RS))

    # equals 1 when crossing BBAnds
    over = ((Stock_RSI['RSI'] - Top_BBand).apply(lambda x: x > 0))
    under = ((Stock_RSI['RSI'] - Lower_BBand).apply(lambda x: x < 0))

    Stock_RSI['Buy/Sell'] = under + (-1) * over

    buy_sell(Stock_RSI, portfolio_initial_value, commission, stop_loss_percentage)

    buy_RSI = Stock_RSI[Stock_RSI['Buy or Sell'] == 1]
    sell_RSI = Stock_RSI[Stock_RSI['Buy or Sell'] == -1]

    start = Stock_RSI.index[0].date()
    end = Stock_RSI.index[len(Stock_RSI['Close']) - 1].date()

    # PLOTS################################################
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    Stock_RSI['Close'].plot(label='Close', color='k')
    if buy_RSI.empty == 0:
        plt.plot(buy_RSI.index, buy_RSI['Close'], marker='^', markersize=10, linestyle="None", color='g', label='Buy')
    if sell_RSI.empty == 0:
        plt.plot(sell_RSI.index, sell_RSI['Close'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.title('{} Closing Price'.format(stock_name))
    plt.ylabel('Share Price')
    plt.legend(loc='upper left')

    plt.subplot(312)
    plt.title('{} CLosing Price with Relative Strength Index'.format(stock_name))
    Stock_RSI['RSI'].plot(label='RSI', color='k')
    plt.hlines(Top_BBand, start, end, color='b', linestyles='--')
    plt.hlines(50, start, end, color='Grey', linestyles='--')
    plt.hlines(Lower_BBand, start, end, color='b', linestyles='--')
    if buy_RSI.empty == 0:
        plt.plot(buy_RSI.index, buy_RSI['RSI'], marker='^', markersize=10, linestyle="None", color='g', label='Buy')
    if sell_RSI.empty == 0:
        plt.plot(sell_RSI.index, sell_RSI['RSI'], marker='v', markersize=10, linestyle="None", color='r', label='Sell')
    plt.ylim((10, 90))
    plt.legend(loc='upper left')

    plt.subplot(313)
    plt.plot(Stock_RSI['Portfolio'], label='Portfolio Value', color='k')
    plt.title('Portfolio Value')
    if buy_RSI.empty == 0:
        plt.plot(buy_RSI.index, buy_RSI['Portfolio'], marker='^', markersize=10, linestyle="None", color='g',
                 label='Buy')
    if sell_RSI.empty == 0:
        plt.plot(sell_RSI.index, sell_RSI['Portfolio'], marker='v', markersize=10, linestyle="None", color='r',
                 label='Sell')
    plt.xlabel('Date')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def date_select():
    run = True
    while run:
        date_run = input(
            "Do you want the date range to be a custom date set or have today as last analysis day (custom/today): ")
        date_run = date_run.lower()
        if date_run == 'today':
            start, end = today_start()
            run = False
        elif date_run == 'custom':
            start, end = custom_start()
            run = False
        else:
            print('Value entered incorrectly')
    return start, end


def today_start():
    end = datetime.date.today()
    while True:
        len_dates = input("How many months would you like to analyze: ")
        if len_dates.isnumeric():
            start = datetime.date.today() - datetime.timedelta(days=float(len_dates) * 30)
            break
        else:
            print('Non-numeric value entered')
    return start, end


def custom_start():
    while True:
        start = input("Enter custom start date YYYY/MM/DD: ")
        end = input("Enter custom end date YYYY/MM/DD: ")
        dates_end = end.split('/')
        dates_start = start.split('/')
        if start > end:
            print('Start date comes after end date, reenter dates')
        elif datetime.date(dates_end[0], dates_end[1], dates_end[2]) > datetime.date.today():
            print('End date is in the future, reenter dates')
        elif len(start) != 10 or len(end) != 10:
            print('Incorrect date format')
        elif 1 >= dates_start[1] >= 13 and 1 >= dates_end[1] >= 13:
            print('Dates entered incorrectly')
        else:
            break
    return start, end


def commission_set():
    while True:
        comm = input("Enter the stock broker commission rate: ")
        if comm.isnumeric():
            commission = float(comm)
            break
        else:
            print('A non-numeric value was entered please')
    return commission


def port_initial_val():
    while True:
        port = input("Enter the initial portfolio value: ")
        if port.isnumeric():
            portfolio_initial_value = float(port)
            break
        else:
            print('A non-numeric value was entered please')
    return portfolio_initial_value


def stop_loss_p():
    while True:
        stop_loss = input("Enter the stop loss percentage, if no stop loss trigger to be used enter 0: ")
        if stop_loss.isnumeric():
            stop_loss_percentage = float(stop_loss)
            break
        else:
            print('A non-numeric value was entered please')
    return stop_loss_percentage


def data_select():
    run = True
    while run:
        stockname = input("Please enter the stock ticker name you would like to analyze: ")
        start, end = date_select()
        commission = commission_set()
        portfolio_initial_value = port_initial_val()
        stop_loss_percentage = stop_loss_p()
        try:
            stock_dataframe = data.DataReader(stockname.upper(), 'google', start, end)
            run = False
        except:
            print('Stock ticker name is invalid please reenter')
    return stockname.upper(), stock_dataframe, portfolio_initial_value, commission, stop_loss_percentage


def stock_analytics_select(stock_name, stock, port_value, comm, s_l_p):
    while True:
        tool = input("Which analysis tool would you like to use: SMA, MACD, Bollinger Bands, RSI: ")
        if tool.lower() == 'sma':
            sma(stock_name, stock, port_value, comm, s_l_p)
            break
        elif tool.lower() == 'macd':
            macd(stock_name, stock, port_value, comm, s_l_p)
            break
        elif tool.lower() == 'bollinger bands' or tool.lower() == 'bbands':
            bbands(stock_name, stock, port_value, comm, s_l_p)
            break
        elif tool.lower() == 'rsi':
            rsi(stock_name, stock, port_value, comm, s_l_p)
            break
        else:
            print('Stock analysis tool entered incorrectly')


def stock_analytics_run():
    first_run = True
    while True:
        if first_run:
            stockname, stock, port_value, comm, s_l_p = data_select()
            stock_analytics_select(stockname, stock, port_value, comm, s_l_p)
            first_run = False
        elif not first_run:
            run_again = input("Would you like to analyze the same stock using a different analysis tool? (yes/no)")
            if run_again.lower() == 'yes' or run_again.lower() == 'y':
                stock_analytics_select(stockname, stock, port_value, comm, s_l_p)
            else:
                diff_stock = input(
                    "Would you like to analyze the a different stock using the same portfolio, commission and stop loss values? (yes/no)")
                if diff_stock == 'yes' or diff_stock == 'y':
                    stockname = input("Please enter the stock ticker name you would like to analyze: ")
                    stock_analytics_select(stockname, stock, port_value, comm, s_l_p)
                else:
                    break


stock_analytics_run()
