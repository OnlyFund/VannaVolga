import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from lognormal_sabr import Hagan2002_LogNormal as log_sabr
from matplotlib.ticker import MultipleLocator
from lognormal_sabr import lognormal_vol as log_vol
import Black76
sns.set()

"This script is used to implement SABR Log-Normal/Normal volsurface model and plot smile/surface using calibrate_and_plot functions" \
"To use these functions the user need either to specify strikes/vols to calibrate the stochastic parameters (alpha,beta,rho,nu) or the parameters themselves." \
"Also required are the forward rates, forward term (in years), strike range to plot,strike_shift, and normal ATM volatility"

"In addition to this script, there are also additional functions to convert log-normal/shifted log-normal/normal vols, as well as call option pricing using" \
"either Black76 pricing model or Bachalier model (for normal vol)"

"Parameters description:" \
"alpha = the level of atm vol" \
"beta = contorls the skewness and a CEV exponent" \
"rho = controls the spot-vol correlation" \
"nu = the volatility-of-volatility" \
"shift= in order to handle a transition between positive and negative rates (and vice versa) a shift is necessary to avoid erros in log-function" \
"v_atm_norm = the normal ATM volatility (represented in number, not % terms) "



def calibrate_and_plot_smile(forward_rate,shift,t,v_atm_n,beta,strikes_calib,strike_range,vols,plot_vol='LN'):
    #forward rate = the atm/forward rate for the term
    #shift = the shift applied for close-to-zero rates, shift=2 is recommended
    #t = maturity term
    #v_atm_n = the atm normal volatility
    #beta = beta parameter, beta=0.5 recommended
    #strike_calib = the strikes used for parameters calibration
    #vols = the corresponded vols used for calibration
    #plot_vol = the smile to plot , can take either 'LN' for log-normal(default), or 'Normal' for normal vol

    calib = log_sabr(f=forward_rate,shift=shift,t=t,v_atm_n=v_atm_n,beta=beta).fit(strikes_calib,vols)

    smile = [log_sabr(f=forward_rate,
                      shift=shift,
                      t=t,v_atm_n=v_atm_n,
                      beta=beta,rho=calib[1],volvol=calib[2]).lognormal_vol(strike)*100 for strike in strike_range]

    normal_smile = [Black76.hagan_lognormal_to_normal(strike,forward_rate,shift,t,vol/100)*100 for vol,strike in zip(smile,strike_range)]

    fig, ax = plt.subplots()

    if plot_vol=='LN':
        ax.plot(smile)
        plt.title('Log-Normal Vol Plot',fontsize=9)


    else:
        ax.plot(normal_smile)
        plt.title('Normal Vol Plot',fontsize=12)

    plt.xticks(np.arange(len(strike_range)),strike_range,rotation=45,fontsize=8,color='navy')
    plt.yticks(fontsize=8,color='navy')
    ax.xaxis.set_major_locator(MultipleLocator(len(strike_range)/10))
    ax.set_xlabel('Strikes',fontsize=9)
    ax.set_ylabel('Volatility',fontsize=9)
    plt.tight_layout()
    plt.show()




#An example for calibrate_and_plot_smile function
strikes_calib = np.array([-0.47533,1.02467,2.02467,3.52467,4.02467])
strike_range_1 = np.around(np.linspace(-1,5,100),3)
vol_calib = np.array([33.93,22.28,23.03,27.22,28.51])
forward_rate = 1.52467
shift= 2
t = 4
beta = 0.5
v_atm_n = 0.7767

calibrate_and_plot_smile(forward_rate=forward_rate,
                                   shift=shift,
                                   v_atm_n=v_atm_n,
                                   beta=beta,
                                   strikes_calib=strikes_calib,
                                   strike_range=strike_range_1
                                   ,vols=vol_calib,t=t,plot_vol='Normal')


def calibrate_and_plot_surface(forward_curve, forward_term,
                               strike_calibration, vol_calibration, strike_range,
                               normal_vol_term, shift, beta):

    #This function is used for calibrating multiple smiles (aka, volatility surface)
    #Similar to calibrate_and_plot_smile the required parameters, but one should enter the parameters
    #for multiple expiries/terms


    normal_vol_term = np.divide(normal_vol_term, 100)

    strike_vol_calibrate, data = [], []
    swaption_surface, normal_surface = [], []

    strikes_calibrate = [strike_calibration + forward for forward in forward_curve]

    vol_calibration = vol_calibration.reshape((len(forward_term), len(strike_calibration)), order='c')

    for term in forward_term:
        for strike, vol in zip(strikes_calibrate, vol_calibration):
            data = [strike, vol]
            strike_vol_calibrate.append(data)

    for term in range(0, len(forward_term)):
        calib = log_sabr(f=forward_curve[term], shift=shift, t=forward_term[term], v_atm_n=normal_vol_term[term],
                         beta=beta).fit(strike_vol_calibrate[term][0], strike_vol_calibrate[term][1])


        for strike in strike_range:
            swaption_surface.append(
                log_sabr(f=forward_curve[term], shift=shift, t=forward_term[term], v_atm_n=normal_vol_term[term],
                         beta=beta,
                         rho=calib[1], volvol=calib[2]).lognormal_vol(strike + forward_curve[term]) * 100)

            normal_surface.append(Black76.hagan_lognormal_to_normal(k=(strike + forward_curve[term]),
                                                                    f=forward_curve[term],
                                                                    s=shift,
                                                                    t=forward_term[term],
                                                                    v_sln=swaption_surface[-1] / 100) * 100)

    swaption_surface = np.array(swaption_surface).reshape((len(forward_term), len(strike_range)), order='c')
    normal_surface = np.array(normal_surface).reshape((len(forward_term), len(strike_range)), order='c')

    fig = go.Figure(data=[go.Surface(z=swaption_surface,
                                     x=strike_range,
                                     y=forward_term, colorscale='balance')])
    fig.update_layout(title='Log-Normal VolSurface',
                      autosize=True, width=1000, height=1000,
                      scene=dict(xaxis_title='strikes',
                                 yaxis_title='forward term',
                                 zaxis_title='volatility'
                                 ))

    fig2 = go.Figure(data=[go.Surface(z=normal_surface,
                                      x=strike_range,
                                      y=forward_term, colorscale='balance')])

    fig2.update_layout(title='Normal VolSurface',
                       autosize=True, width=1000, height=1000,
                       scene=dict(xaxis_title='strikes',
                                  yaxis_title='forward term',
                                  zaxis_title='volatility'
                                  ))
    fig.show()
    fig2.show()


forward_curve = np.array([1.244,1.385,1.445,1.757,2.2])
forward_term = np.array([1,2,3,4,5])
strike_calibration = np.linspace(-2.5,2.5,6)
strike_range = np.around(np.linspace(-2.5,2.5,30),3)
vol_calibration = np.array([40.73,25.85,22.68,25.56,28.84,31.53,
                           39.84,26.55,22.62,24,26.41,28.58,
                           40.95,27.15,22.51,23.71,26.28,28.63,
                           42.22,27.77,22.37,23.36,26.09,28.62,43.36,28.40,22.29,23.00,25.88,28.59])

v_atm_n = [76.92,78.34,78.39,77.59,77.12]


#Example of calibrate_and_plot_surface function
calibrate_and_plot_surface(forward_curve,forward_term,strike_calibration,vol_calibration,strike_range,v_atm_n,shift,beta)


alpha = 0.05
beta = 0.5
rho = -0.3
nu = 0.5
f = 2.51
strike_range = np.linspace(-2.5,2.5,100)
t = 1

def plot_smile_from_parameters(f,strike_range,t,alpha,beta,rho,nu):
#plotting SABR smile from given parameters.
#Note that this function CANNOT handle both -tive and +tive strikes.

    strike_range = np.around(strike_range+ f,2)
    vol_smile = []

    for strike in strike_range:
        vol_ln = log_vol(strike,f,t,alpha,beta,rho,nu)
        vol_smile.append(vol_ln*100)

    fig,ax = plt.subplots()

    ax.plot(vol_smile)
    plt.title('Log Normal Volatility Smile, calibrated from parameters')

    plt.xticks(np.arange(len(strike_range)), strike_range, rotation=45, fontsize=8, color='navy')
    plt.yticks(fontsize=8, color='navy')
    ax.xaxis.set_major_locator(MultipleLocator(len(strike_range) / 10))
    ax.set_xlabel('Strikes', fontsize=9)
    ax.set_ylabel('Volatility', fontsize=9)
    plt.tight_layout()
    plt.show()

#Example of plot_smile_from_parameters
plot_smile_from_parameters(f,strike_range,t,alpha,beta,rho,nu)

