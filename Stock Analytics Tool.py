from pandas_datareader import data
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


# Buy/sell signal
def buy_sell(data_frame, portfolio_value, commission, stop_loss):
    buying_power = portfolio_value
    if not stop_loss or stop_loss == 0:
        stop_loss_percent = 0
    else:
        stop_loss_percent = 1 - (stop_loss / 100)
    buy_price = 0
    shares = 0
    value = 0
    for i in data_frame.index.values:
        data_frame.set_value(i, 'Own', value)
        data_frame.set_value(i, 'Shares', shares)
        data_frame.set_value(i, 'Buying Power', buying_power)
        port_val = data_frame.ix[i]['Buying Power'] + data_frame.ix[i]['Shares'] * data_frame.ix[i]['Open']
        data_frame.set_value(i, 'Portfolio', port_val)
        if data_frame.ix[i]['Buy/Sell'] == 1 and data_frame.ix[i]['Own'] == 0 and (
                    commission + data_frame.ix[i]['Open']) < data_frame.ix[i]['Buying Power']:
            value = 1
            # 2 times commission subtracted so that there is enough to the sell the shares
            buying_power -= 2 * commission
            shares = math.floor(buying_power / data_frame.ix[i]['Open'])
            buying_power = buying_power - shares * data_frame.ix[i]['Open'] + commission
            data_frame.set_value(i, 'Buy or Sell', 1)
            buy_price = data_frame.ix[i]['Open']
        elif data_frame.ix[i]['Buy/Sell'] == -1 and data_frame.ix[i]['Own'] == 1 and commission < \
                data_frame.ix[i][
                    'Buying Power']:
            value = 0
            buying_power -= commission
            buying_power = buying_power + shares * data_frame.ix[i]['Open']
            shares = 0
            data_frame.set_value(i, 'Buy or Sell', -1)
        # Stoploss condition
        if data_frame.ix[i]['Close'] <= buy_price * stop_loss_percent and data_frame.ix[i][
            'Own'] == 1 and commission < \
                data_frame.ix[i]['Buying Power']:
            value = 0
            buying_power -= commission
            buying_power = buying_power + shares * data_frame.ix[i]['Open']
            shares = 0
            data_frame.set_value(i, 'Buy or Sell', -1)

    data_frame.drop('Buy/Sell', axis=1, inplace=True)


# Analysis tools
def sma(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    fast, slow = sma_sma_values()
    stock__s_m_a = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power', 'Portfolio'])
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

    buy_sell_sig = [buy__s_m_a, sell__s_m_a]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    label = '{} Close'.format(stock_name.upper())
    close_plot(stock__s_m_a, buy_sell_sig, 'Close', label, 'k')

    plt.subplot(312)
    stock__s_m_a['Fast SMA'].plot(label='{} Day Avg'.format(fast), color='m')
    label = '{} Day Avg'.format(slow)
    signal_plot(stock__s_m_a, buy_sell_sig, 'Slow SMA', label, 'c')
    plt.title('{} - {} Day Moving Average and {} Day Moving Average'.format(stock_name.upper(), fast, slow))

    plt.subplot(313)
    port_plot(stock__s_m_a, buy_sell_sig, 'Portfolio', 'Portfolio Value', 'k')

    plt.tight_layout()
    plt.show()


def macd(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    f_ema, s_ema, sig_ema = macd_ema_values()

    stock__m_a_c_d = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power', 'Portfolio'])
    stock__m_a_c_d = stock__m_a_c_d.join(new_col)

    ema_f = 2 / (f_ema + 1) * (stock__m_a_c_d['Close'][f_ema - 1]) + (1 - 2 / (f_ema + 1)) * stock__m_a_c_d['Close'][
                                                                                             0:f_ema].mean()
    ema_s = 2 / (s_ema + 1) * (stock__m_a_c_d['Close'][s_ema - 1]) + (1 - 2 / (s_ema + 1)) * stock__m_a_c_d['Close'][
                                                                                             0:s_ema].mean()

    ema_f_list = []
    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i < f_ema - 1:
            ema_f_list.append(np.nan)
        else:
            ema_f_list.append(ema_f)
            ema_f = (2 / (f_ema + 1)) * stock__m_a_c_d['Close'][i] + (1 - 2 / (f_ema + 1)) * ema_f

    ema_s_list = []
    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i < s_ema - 1:
            ema_s_list.append(np.nan)
        else:
            ema_s_list.append(ema_s)
            ema_s = (2 / (s_ema + 1)) * stock__m_a_c_d['Close'][i] + (1 - 2 / (s_ema + 1)) * ema_s

    stock__m_a_c_d['MACD'] = np.array(ema_f_list) - np.array(ema_s_list)
    signal_val = 2 / (sig_ema + 1) * (stock__m_a_c_d['MACD'][s_ema + sig_ema]) + (1 - 2 / (sig_ema + 1)) * \
                                                                                 stock__m_a_c_d['MACD'][
                                                                                 s_ema:s_ema + sig_ema].mean()
    signal = []

    for i in range(0, len(stock__m_a_c_d['Close'])):
        if i < s_ema + sig_ema:
            signal.append(np.nan)
        else:
            signal.append(signal_val)
            signal_val = (2 / (sig_ema + 1)) * stock__m_a_c_d['MACD'][i] + (1 - 2 / (sig_ema + 1)) * signal_val

    stock__m_a_c_d['Signal'] = signal
    stock__m_a_c_d['MACD_Hist'] = (stock__m_a_c_d['MACD'] - stock__m_a_c_d['Signal'])

    inc_dec__m_a_c_d = np.sign(stock__m_a_c_d['MACD_Hist'])
    stock__m_a_c_d['Buy/Sell'] = np.sign(inc_dec__m_a_c_d.diff())

    buy_sell(stock__m_a_c_d, portfolio_initial_value, commission, stop_loss_percentage)
    buy__m_a_c_d = stock__m_a_c_d[stock__m_a_c_d['Buy or Sell'] == 1]
    sell__m_a_c_d = stock__m_a_c_d[stock__m_a_c_d['Buy or Sell'] == -1]

    buy_sell_sig = [buy__m_a_c_d, sell__m_a_c_d]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))

    plt.subplot(311)
    label = '{} Close'.format(stock_name.upper())
    close_plot(stock__m_a_c_d, buy_sell_sig, 'Close', label, 'k')

    plt.subplot(312)
    stock__m_a_c_d['MACD'].plot(label='MACD', color='m')
    label = 'Signal'
    signal_plot(stock__m_a_c_d, buy_sell_sig, 'Signal', label, 'c')
    plt.title('{} - MACD and Signal'.format(stock_name.upper()))

    plt.subplot(313)
    port_plot(stock__m_a_c_d, buy_sell_sig, 'Portfolio', 'Portfolio Value', 'k')

    plt.tight_layout()
    plt.show()


def bbands(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    ave_window, std_t = bbands_values()
    stock__b_bands = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power', 'Portfolio'])
    stock__b_bands = stock__b_bands.join(new_col)

    h_l_c = (stock__b_bands['High'] + stock__b_bands['Low'] + stock__b_bands['Close']) / 3

    twenty_day_avg = h_l_c.rolling(window=ave_window).mean()
    twenty_day_std = h_l_c.rolling(window=ave_window).std()

    # Calculating Upper and Lower Band
    upp_band = twenty_day_avg + std_t * twenty_day_std
    low_band = twenty_day_avg - std_t * twenty_day_std

    # Buy and Sell indicators, based on when above or below Bollinger Bands
    over = ((stock__b_bands['Close'] - upp_band).apply(lambda x: x > 0))
    under = ((stock__b_bands['Close'] - low_band).apply(lambda x: x < 0))

    stock__b_bands['Buy/Sell'] = under + (-1) * over
    buy_sell(stock__b_bands, portfolio_initial_value, commission, stop_loss_percentage)

    buy__b_bands = stock__b_bands[stock__b_bands['Buy or Sell'] == 1]
    sell__b_bands = stock__b_bands[stock__b_bands['Buy or Sell'] == -1]

    buy_sell_sig = [buy__b_bands, sell__b_bands]

    # PLOTS################################################
    plt.figure(figsize=(15, 10))

    plt.subplot(311)
    label = '{} Close'.format(stock_name.upper())
    close_plot(stock__b_bands, buy_sell_sig, 'Close', label, 'k')

    plt.subplot(312)
    upp_band.plot(label='Upper Band', color='b', linestyle='--')
    low_band.plot(label='Lower Band', color='b', linestyle='--')
    label = '{} Close'.format(stock_name.upper())
    signal_plot(stock__b_bands, buy_sell_sig, 'Close', label, 'k')
    plt.title('{} - Bollinger Bands'.format(stock_name.upper()))

    plt.subplot(313)
    port_plot(stock__b_bands, buy_sell_sig, 'Portfolio', 'Portfolio Value', 'k')
    plt.tight_layout()
    plt.show()


def rsi(stock_name, stock, portfolio_initial_value, commission, stop_loss_percentage):
    top__b_band, lower__b_band, interval = rsi_values()

    stock__r_s_i = stock.copy()
    new_col = pd.DataFrame(columns=['Buy/Sell', 'Buy or Sell', 'Own', 'Shares', 'Buying Power', 'Portfolio'])
    stock__r_s_i = stock__r_s_i.join(new_col)

    # Determining gains and losses on daily basis. Binning gains and losses
    diff_close = stock__r_s_i['Close'].diff()
    rs_pos = diff_close.apply(lambda x: x > 0) * diff_close
    rs_neg = diff_close.apply(lambda x: x < 0) * diff_close * -1

    rs_positive = rs_pos[0:interval].sum() / interval
    rs_negative = rs_neg[0:interval].sum() / interval

    # Smoothed Average Gain and Average Loss calculation
    rsp = []
    rsn = []
    for i in range(0, len(stock__r_s_i['Close'])):
        if i > 13:
            rsp.append(rs_positive)
            rsn.append(rs_negative)
            rs_positive = (rs_positive * 13 + rs_pos[i]) / interval
            rs_negative = (rs_negative * 13 + rs_neg[i]) / interval
        else:
            rsp.append(0)
            rsn.append(0)

    r_s = np.array(rsp) / np.array(rsn)
    stock__r_s_i['RSI'] = 100 - (100 / (1 + r_s))

    # equals 1 when crossing BBAnds
    over = ((stock__r_s_i['RSI'] - top__b_band).apply(lambda x: x > 0))
    under = ((stock__r_s_i['RSI'] - lower__b_band).apply(lambda x: x < 0))

    stock__r_s_i['Buy/Sell'] = under + (-1) * over

    buy_sell(stock__r_s_i, portfolio_initial_value, commission, stop_loss_percentage)

    buy__r_s_i = stock__r_s_i[stock__r_s_i['Buy or Sell'] == 1]
    sell__r_s_i = stock__r_s_i[stock__r_s_i['Buy or Sell'] == -1]

    buy_sell_sig = [buy__r_s_i, sell__r_s_i]

    start = stock__r_s_i.index[0].date()
    end = stock__r_s_i.index[len(stock__r_s_i['Close']) - 1].date()

    # PLOTS################################################
    plt.figure(figsize=(15, 10))

    plt.subplot(311)
    label = '{} Close'.format(stock_name.upper())
    close_plot(stock__r_s_i, buy_sell_sig, 'Close', 'Close', 'k')

    plt.subplot(312)
    label = '{} Close'.format(stock_name.upper())
    signal_plot(stock__r_s_i, buy_sell_sig, 'RSI', label, 'k')
    plt.hlines(top__b_band, start, end, color='b', linestyles='--')
    plt.hlines((top__b_band + lower__b_band) / 2, start, end, color='Grey', linestyles='--')
    plt.hlines(lower__b_band, start, end, color='b', linestyles='--')
    plt.title('{} Closing Price with Relative Strength Index'.format(stock_name.upper()))

    plt.subplot(313)
    port_plot(stock__r_s_i, buy_sell_sig, 'Portfolio', 'Portfolio', 'k')
    plt.tight_layout()
    plt.show()


# Plot creation
def plot_decorator(function):
    def signals(dataframe, tool, col, label, color):
        function(dataframe, tool, col, label, color)
        print()
        if tool[0].empty == 0:
            plt.plot(tool[0].index, tool[0][col], marker='^', markersize=10, linestyle="None",
                     color='g', label='Buy')
        if tool[1].empty == 0:
            plt.plot(tool[1].index, tool[1][col], marker='v', markersize=10, linestyle="None",
                     color='r', label='Sell')
        plt.xlabel('Date')
        plt.legend(loc='upper left')
    return signals


@plot_decorator
def close_plot(dataframe, tool, col, label, color):
    dataframe[col].plot(label=label, color=color)
    plt.title(label[:-1] + 'ing Price')
    plt.ylabel('Share Price')


@plot_decorator
def signal_plot(dataframe, tool, col, label, color):
    dataframe[col].plot(label=label, color=color)


@plot_decorator
def port_plot(dataframe, tool, col, label, color):
    dataframe[col].plot(label=label, color=color)
    plt.title('Portfolio')
    plt.ylabel('Portfolio Value')


# Initial data selection
def port_initial_val():
    while True:
        try:
            portfolio_initial_value = int(input("Please enter your initial portfolio value with no decimal values: "))
        except ValueError:
            print('Value was entered incorrectly')
        if portfolio_initial_value<0:
            print('Portfolio value cannot be negative')
        else:
            break
    return portfolio_initial_value


def commission_set():
    while True:
        comm = input("Enter the stock broker's commission fee. Leave blank if no stop loss trigger to be used: ")
        if not comm.split():
            comm = 0
            break
        else:
            try:
                comm = float(comm)
                break
            except ValueError:
                print('A non-numeric value was entered')
    return comm


def stop_loss_p():
    stop_loss_percentage = []
    while True:
        stop_loss = input("Enter the stop loss percentage. Leave blank if no stop loss trigger to be used: ")
        if not stop_loss.split():
            break
        else:
            try:
                stop_loss_percentage = float(stop_loss)
                break
            except ValueError:
                print('A non-numeric value was entered')
    return stop_loss_percentage


def date_select():
    while True:
        date_run = input(
            "To use today as the last analysis day leave blank, otherwise type 'custom' to select exact dates: ")
        if not date_run.split():
            start, end = today_start()
            break
        elif date_run.lower() == 'custom':
            start, end = custom_start()
            break
        else:
            print('Value was entered incorrectly')
    return start, end


def today_start():
    end = datetime.date.today()
    while True:
        len_dates = input("How many months would you like to analyze: ")
        if len_dates.isnumeric():
            start = datetime.date.today() - datetime.timedelta(days=float(len_dates) * 30)
            break
        else:
            print('A non-numeric value was entered')
    return start, end


def custom_start():
    while True:
        start = input("Enter the custom start date as YYYY-MM-DD: ")
        try:
            dates_start = list(map(int, start.split('-')))
            if len(start) != 10:
                print('Incorrect date format')
            elif 1 > dates_start[1] > 12 or 1 > dates_start[2] > 31:
                print('Date entered incorrectly')
            else:
                break
        except ValueError:
            print('Date entered incorrectly')
    while True:
        end = input("Enter the custom end date as YYYY-MM-DD: ")
        try:
            dates_end = list(map(int, end.split('-')))
            if start > end:
                print('Start date comes after end date, please reenter dates')
                custom_start()
            elif datetime.date(dates_end[0], dates_end[1], dates_end[2]) > datetime.date.today():
                print('End date is in the future, reenter dates')
            elif len(end) != 10:
                print('Incorrect date format')
            elif 1 > dates_end[1] > 12 or 1 > dates_end[2] > 31:
                print('Dates entered incorrectly')
            else:
                break
        except ValueError:
            print('Date entered incorrectly')
    return start, end


def stock_data_select(start, end):
    while True:
        stockname = input("Please enter the stock ticker symbol: ")
        try:
            stock_dataframe = data.DataReader(stockname.upper(), 'google', start, end)
            break
        except ValueError:
            print('Stock ticker name is invalid please reenter')
    return stockname.upper(), stock_dataframe


# Analysis tool selection
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
            print('Stock analysis tool was entered incorrectly')


# Analysis tool values
def sma_sma_values():
    fast = 5
    slow = 20
    while True:
        sma_custom = input("To use 5,20 day moving average values press enter, otherwise enter 'custom': ")
        if not sma_custom.split():
            break
        elif sma_custom.lower() == 'custom':
            while True:
                fast = input("Please enter fast moving average day count: ")
                try:
                    fast = int(fast)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                slow = input("Please enter slow moving average day count: ")
                try:
                    slow = int(slow)
                    if fast > slow:
                        print('The fast moving average is greater than the slow moving average')
                    else:
                        break
                except ValueError:
                    print('Value entered incorrectly')
            break
        else:
            print('Value entered incorrectly')
    return fast, slow


def macd_ema_values():
    fast_ema = 12
    slow_ema = 26
    signal_ema = 9
    while True:
        ema_custom = input(
            "To use 12,26,9 day exponential moving average values press enter, otherwise enter 'custom': ")
        if not ema_custom.split():
            break
        elif ema_custom.lower() == 'custom':
            while True:
                fast_ema = input("Please enter the fast exponential moving average day count: ")
                try:
                    fast_ema = int(fast_ema)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                slow_ema = input("Please enter the slow moving average day count: ")
                try:
                    slow_ema = int(slow_ema)
                    if fast_ema > slow_ema:
                        print('The fast exponential moving average is greater than the slow exponential moving average')
                        macd_ema_days()
                    else:
                        break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                signal_ema = input("Please enter the signal exponential moving average day count: ")
                try:
                    signal_ema = int(signal_ema)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            break
        else:
            print('Value entered incorrectly')
    return fast_ema, slow_ema, signal_ema


def bbands_values():
    window = 20
    std_t = 2
    while True:
        bbands_custom = input(
            "To use 20 day moving average and 2 * standard deviation press enter, otherwise enter 'custom': ")
        if not bbands_custom.split():
            break
        elif bbands_custom.lower() == 'custom':
            while True:
                window = input("Please enter the moving average day count: ")
                try:
                    window = int(window)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                std_t = input("Please enter number of standard deviations to use in band calculation: ")
                try:
                    std_t = int(std_t)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            break
        else:
            print('Value entered incorrectly')
    return window, std_t


def rsi_values():
    top_band = 70
    lower_band = 30
    interval = 14
    while True:
        rsi_custom = input(
            "To use 70 and 30 as the values of the top and bottom bands and 14 days for the interval press enter, otherwise enter 'custom': ")
        if not rsi_custom.split():
            break
        elif rsi_custom.lower() == 'custom':
            while True:
                top_band = input("Please enter the value for the upper band: ")
                try:
                    top_band = int(top_band)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                lower_band = input("Please enter the value for the lower band: ")
                try:
                    lower_band = int(lower_band)
                    if lower_band > top_band:
                        print('Bottom band is set to a higher value than the top band')
                        rsi_values()
                    break
                except ValueError:
                    print('Value entered incorrectly')
            while True:
                interval = input("Please enter the interval day count: ")
                try:
                    interval = int(interval)
                    break
                except ValueError:
                    print('Value entered incorrectly')
            break
        else:
            print('Value entered incorrectly')
    return top_band, lower_band, interval


# Putting it together
def stock_analytics_run():
    port_value = port_initial_val()
    comm = commission_set()
    s_l_p = stop_loss_p()
    start, end = date_select()
    stockname, stock = stock_data_select(start, end)
    first_run = True
    while True:
        if first_run:
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
                    stockname, stock = stock_data_select(start, end)

                    stock_analytics_select(stockname, stock, port_value, comm, s_l_p)
                else:
                    break


stock_analytics_run()
