import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from ipywidgets import interact
import ipywidgets as widgets

class ExponentialPlot():

    @staticmethod
    def exponential_interact():
        interact(ExponentialPlot.exponential_pdf_plot,
            lambd=widgets.FloatSlider(min=0.5, max=5, step=0.1, value=1),
        );

    @staticmethod
    def exponential_pdf_plot(lambd):
        """
        lambd = rate parameter
        """
        pdf = lambda x,lambd: lambd*np.exp(-lambd*x)
        mean = 1/lambd
        std = mean

        upper_x = expon.ppf(0.95,lambd)
        X = np.arange(0.1,upper_x,upper_x/1000)
        fx = [pdf(x,lambd) for x in X]

        plt.plot(X,fx, label='mean={:.2f}, std={:.2f}'.format(mean,std));
        plt.title('Exponential probability distribution function')
        plt.xlabel('x')
        plt.ylabel('pdf(x)')
        plt.legend()
        plt.grid()
        plt.show();

    @staticmethod
    def exponential_pdf_cdf_plots():
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
        #ax = fig.add_subplot(111)

        x = np.linspace(0,3,1000)
        exp_pdf = lambda x,lambd: lambd*np.exp(-lambd*x)
        exp_cdf = lambda x,lambd: 1 - np.exp(-lambd*x)

        ax1.plot(x,exp_pdf(x,0.5),label='lamb = 0.5');
        ax1.plot(x,exp_pdf(x,1),label='lamb = 1');
        ax1.plot(x,exp_pdf(x,2),label='lamb = 2');
        ax1.plot(x,exp_pdf(x,4),label='lamb = 4');
        ax1.plot(x,exp_pdf(x,8),label='lamb = 8');
        ax1.set_xlabel('x')
        ax1.set_ylabel('pdf(x)')
        ax1.legend()
        ax1.grid()

        ax2.plot(x,exp_cdf(x,0.5),label='lamb = 0.5');
        ax2.plot(x,exp_cdf(x,1),label='lamb = 1');
        ax2.plot(x,exp_cdf(x,2),label='lamb = 2');
        ax2.plot(x,exp_cdf(x,4),label='lamb = 4');
        ax2.plot(x,exp_cdf(x,8),label='lamb = 8');
        ax2.set_xlabel('x')
        ax2.set_ylabel('cdf(x)')
        ax2.legend()
        ax2.grid()

        plt.show()
