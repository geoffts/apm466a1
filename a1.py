import copy
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

#-------------------- getting data -------------------------

def csv_to_list(csv_file) -> list[list[str]]:
    data = []
    for line in open(csv_file, 'r'):
        data.append(line.strip().split(','))
    return data

#-------------------- getting dirty price -------------------------

def append_last_payment(data: list[list[str]], today: int) -> list[list[str]]:
    # here data is the data and today is the date the price was collected,
    # where today is an integer from 0 to 9, representing Jan 6 to Jan 17

    new_data = copy.deepcopy(data)
    for n in range(len(new_data)):
        new_data[n].append(0)

    if today >= 0 and today < 5:
        for n in range(len(new_data)):
            new_data[n][15] = 127+today
    else:
        for n in range(len(new_data)):
            new_data[n][15] = 127+2+today
    return new_data

def append_dirty_price(data: list[list[str]], cp: int) -> list[list[str]]:
    # data is the data and cp is the index of the clean price
    # cp must take values from 2 to (and including) 11

    # index of the coupon rate
    cr = 1
    # index of the last payment
    lp = 15

    new_data = copy.deepcopy(data)

    # for each row, calculate and append the dirty price to new_data
    for n in range(len(data)):
        dirty_price = float(data[n][cp])+((float(data[n][lp])/182.5)*(float(data[n][cr])*100)/2)
        new_data[n].append(dirty_price)
    return new_data

#-------------------- yield rates -------------------------

def roots(yield_rate: float, dirty_price: float, num_per: int, coupon_rate: float) -> float:
    # when using this function for fsolve, cannot use math.exp (not sure why, will
    # always result in an error). np.exp on the other hand does work.
    coupon_cashflows = 0
    coupon_payment = coupon_rate*50

    # coupon cashflows
    for n in range(num_per):
        coupon_cashflows = coupon_cashflows + coupon_payment*np.exp(-yield_rate*((n+1)/2))

    known_price = 0
    known_price = coupon_cashflows + 100*np.exp(-yield_rate*(num_per/2))

    return dirty_price - known_price

def yield_rates(data: list[list[str]]) -> list[list[str]]:
    # data is the original data

    # index of the coupon rate
    cr = 1
    # index of the maturity date
    md = 14
    # index of the dirty price
    dp = 16
    # index of the bond name
    name = 0

    yield_rates = []
    ndata = []
    ndata = copy.deepcopy(data)
    ndata = sorted(ndata, key=lambda x: x[md])

    for day in range(10):
        day_data = copy.deepcopy(ndata)
        day_data = append_last_payment(day_data, day)
        day_data = append_dirty_price(day_data, day+2)
        yield_rates_day = []
        for n in range(len(day_data)):
            yield_rate = 0
            yield_rate = fsolve(roots, 0.05, args=(day_data[n][dp],n+1,float(day_data[n][cr])))[0]
            yield_rates_day.append(yield_rate)
        yield_rates.append(yield_rates_day)

    return yield_rates

#-------------------- spot rates -------------------------

def spot_rates(data: list[list[str]]) -> list[list[str]]:
    # data is the fixed data, after appending last payment and dirty price

    # index of the coupon rate
    cr = 1
    # index of the maturity date
    md = 14
    # index of the dirty price
    dp = 16
    # index of the bond name
    name = 0

    spot_rates = []
    ndata = []
    ndata = copy.deepcopy(data)
    ndata = sorted(ndata, key=lambda x: x[md])

    for day in range(10):
        day_data = copy.deepcopy(ndata)
        day_data = append_dirty_price(append_last_payment(day_data,day),day+2)
        spot_rates_day = []
 
        # valuing the first spot_rate (i.e. in six months)
        coupon_payment = 0
        coupon_payment = float(day_data[0][cr])*50
        spot_rate = 0
        spot_rate = -2*math.log((day_data[0][dp]/(coupon_payment+100)))
        spot_rates_day.append(spot_rate)

        # valuing the rest of the spot rates. Indexing at n+1th time period
        for n in range(len(day_data)-1):
            residual = 0
            discounted_cash_flows = 0
            spot_rate = 0
            # the amoung payed at each coupon payment period
            coupon_payment = float(day_data[n+1][cr])*50
            discounted_sum = 0
            for k in range(n):
                discounted_sum = discounted_sum + math.exp(-(spot_rates_day[k]*((k+1)/2)))
            # the following is the sum of the discounted cash flows
            # for the time periods we have already done
            discounted_cash_flows = 0
            discounted_cash_flows = coupon_payment*discounted_sum
            residual = day_data[n+1][dp] - discounted_cash_flows
            if residual <= 0:
                print('Error, ' + day_data[n+1][name] + 'has negative residual')
            else:
                spot_rate = -math.log((residual/(coupon_payment+100)))/((n+1)/2)
                spot_rates_day.append(spot_rate)
        spot_rates.append(spot_rates_day)

    return spot_rates

#-------------------------- forward rates ----------------------

def forward_rates(data: list[list[str]]) -> list[list[str]]:
    # calculates 1 year forward rates for terms from 2-5 years,
    # i.e. 1yr-1yr to 1yr-4yr forward rates.
    
    ndata = copy.deepcopy(data)
    sr = spot_rates(ndata)

    forward_rates = []
    for day in range(10):
        forward_rates_day = []
        for n in range(7):
            forward_rate = 0
            forward_rate = ((sr[day][n+2]*(((n+2)/2)+1))-sr[day][1])/((n+2)/2)
            forward_rates_day.append(forward_rate)
        forward_rates.append(forward_rates_day)

    return forward_rates

#-------------------------- covariance matrices ----------------------

def yield_log_returns(data: list[list[str]]) -> list[list[str]]:
    # returns the log returns of 5 bonds in lists
    ndata = copy.deepcopy(data)
    yr = yield_rates(ndata)

    log_returns = []
    for time in range(5):
        # 2*time is the index of the yield curve we want
        log_returns_time = []
        for day in range(9):
            log_return = math.log((yr[day+1][2*time]/yr[day][2*time]))
            log_returns_time.append(log_return)
        log_returns.append(log_returns_time)

    return log_returns

def yield_covariance_matrix(data: list[list[str]]) -> list[list[str]]:
    ndata = copy.deepcopy(data)
    returns = yield_log_returns(ndata)

    mean_values = []
    for bond in range(5):
        returns_sum = 0
        mean_value = 0
        for r in range(9):
            returns_sum = returns_sum + returns[bond][r]
        mean_value = returns_sum/9
        mean_values.append(mean_value)

    matrix = []
    for bond_1 in range(5):
        row = []
        for bond_2 in range(5):
            covariance = 0
            sample_sum = 0
            for r in range(9):
                sample_sum = sample_sum + (returns[bond_1][r]-mean_values[bond_1])*(returns[bond_2][r]-mean_values[bond_2])
            covariance = sample_sum/8
            row.append(covariance)
        matrix.append(row)

    return matrix

def forward_log_returns(data: list[list[str]]) -> list[list[str]]:
    # returns the log returns of 1yr-1yr, 1yr-2yr, 1yr-3yr, and 1yr-4yr in lists
    ndata = copy.deepcopy(data)
    fr = forward_rates(ndata)

    log_returns = []
    for rate in range(4):
        # 2*rate is the index of the rate we want
        log_returns_rate = []
        for day in range(9):
            log_return = math.log((fr[day+1][2*rate]/fr[day][2*rate]))
            log_returns_rate.append(log_return)
        log_returns.append(log_returns_rate)

    return log_returns

def forward_covariance_matrix(data: list[list[str]]) -> list[list[str]]:
    ndata = copy.deepcopy(data)
    returns = forward_log_returns(ndata)

    mean_values = []
    for rate in range(4):
        returns_sum = 0
        mean_value = 0
        for r in range(9):
            returns_sum = returns_sum + returns[rate][r]
        mean_value = returns_sum/9
        mean_values.append(mean_value)

    matrix = []
    for rate_1 in range(4):
        row = []
        for rate_2 in range(4):
            covariance = 0
            sample_sum = 0
            for r in range(9):
                sample_sum = sample_sum + (returns[rate_1][r]-mean_values[rate_1])*(returns[rate_2][r]-mean_values[rate_2])
            covariance = sample_sum/8
            row.append(covariance)
        matrix.append(row)

    return matrix

#-------------------------- computing eigenvalues and eigenvectors ----------------------

def yield_eigenvalues(data: list[list[str]]) -> list[float]:
    ndata = copy.deepcopy(data)
    matrix = np.array(yield_covariance_matrix(ndata))

    w, v = np.linalg.eig(matrix)

    return w

def yield_eigenvectors(data: list[list[str]]) -> list[float]:
    ndata = copy.deepcopy(data)
    matrix = np.array(yield_covariance_matrix(ndata))

    w, v = np.linalg.eig(matrix)

    return v

def forward_eigenvalues(data: list[list[str]]) -> list[float]:
    ndata = copy.deepcopy(data)
    matrix = np.array(forward_covariance_matrix(ndata))

    w, v = np.linalg.eig(matrix)

    return w

def forward_eigenvectors(data: list[list[str]]) -> list[float]:
    ndata = copy.deepcopy(data)
    matrix = np.array(forward_covariance_matrix(ndata))

    w, v = np.linalg.eig(matrix)

    return v

#---------------- OUTPUTS -----------------------

data = []
data = csv_to_list('data.csv')

yr = []
yr = yield_rates(data)
yrp = []
for day in range(10):
    yrp.append([100*i for i in yr[day]])

sr = []
sr = spot_rates(data)
srp = []
for day in range(10):
    srp.append([100*i for i in sr[day]])

fr = []
fr = forward_rates(data)
frp = []
for day in range(10):
    frp.append([100*i for i in fr[day]])

time_periods = ["Mar '25","Sept '25","Mar '26","Sept '26","Mar '27","Sept '27","Mar '28","Sept '28","Mar '29","Sept '29"]

plt.plot(time_periods,yrp[0],color='b',label='Jan 6th')
plt.plot(time_periods,yrp[1],color='g',label='Jan 7th')
plt.plot(time_periods,yrp[2],color='r',label='Jan 8th')
plt.plot(time_periods,yrp[3],color='c',label='Jan 9th')
plt.plot(time_periods,yrp[4],color='m',label='Jan 10th')
plt.plot(time_periods,yrp[5],color='y',label='Jan 13th')
plt.plot(time_periods,yrp[6],color='k',label='Jan 14th')
plt.plot(time_periods,yrp[7],color='gray',label='Jan 15th')
plt.plot(time_periods,yrp[8],color='orange',label='Jan 16th')
plt.plot(time_periods,yrp[9],color='brown',label='Jan 17th')
plt.legend()
plt.grid()
plt.title('Yield Curve')
plt.xlabel('Date (the first of every stated month)')
plt.ylabel('Yield Rate (in %)')
plt.show()

plt.plot(time_periods,srp[0],color='b',label='Jan 6th')
plt.plot(time_periods,srp[1],color='g',label='Jan 7th')
plt.plot(time_periods,srp[2],color='r',label='Jan 8th')
plt.plot(time_periods,srp[3],color='c',label='Jan 9th')
plt.plot(time_periods,srp[4],color='m',label='Jan 10th')
plt.plot(time_periods,srp[5],color='y',label='Jan 13th')
plt.plot(time_periods,srp[6],color='k',label='Jan 14th')
plt.plot(time_periods,srp[7],color='gray',label='Jan 15th')
plt.plot(time_periods,srp[8],color='orange',label='Jan 16th')
plt.plot(time_periods,srp[9],color='brown',label='Jan 17th')
plt.legend()
plt.grid()
plt.title('Spot Rate Curve')
plt.xlabel('Date (the first of every stated month)')
plt.ylabel('Spot Rate (in %)')
plt.show()

fr_time = ['1yr-1yr','1yr-1.5yr','1yr-2yr','1yr-2.5yr','1yr-3yr','1yr-3.5yr','1yr-4yr',]

plt.plot(fr_time,frp[0],color='b',label='Jan 6th')
plt.plot(fr_time,frp[1],color='g',label='Jan 7th')
plt.plot(fr_time,frp[2],color='r',label='Jan 8th')
plt.plot(fr_time,frp[3],color='c',label='Jan 9th')
plt.plot(fr_time,frp[4],color='m',label='Jan 10th')
plt.plot(fr_time,frp[5],color='y',label='Jan 13th')
plt.plot(fr_time,frp[6],color='k',label='Jan 14th')
plt.plot(fr_time,frp[7],color='gray',label='Jan 15th')
plt.plot(fr_time,frp[8],color='orange',label='Jan 16th')
plt.plot(fr_time,frp[9],color='brown',label='Jan 17th')
plt.legend()
plt.grid()
plt.title('Forward Rate Curve')
plt.xlabel('Start-End')
plt.ylabel('Forward Rate (in %)')
plt.show()

yc = np.array(yield_covariance_matrix(data))
print(yc)
yeva = yield_eigenvalues(data)
print(yeva)
yev = yield_eigenvectors(data)
print(yev)
fc = np.array(forward_covariance_matrix(data))
print(fc)
feva = forward_eigenvalues(data)
print(feva)
fev = forward_eigenvectors(data)
print(fev)
