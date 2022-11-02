
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import random
from scipy.stats import gaussian_kde


class Distribuition:
    
    def __init__(self, sample = None, x = None):
        self.__density = None
        self.__x = x
        self.__sample = sample
        self.__normalized = None
    
    def beta_mix_sim(self, params_list:list, size:int):
        """Generates a random sample of a beta mixture

        Args:
            params_list (list): list of tuples containing the p and beta params. Ex. [(0.5, 100, 50),(0.5, 400, 500)]
            size (int): sample size

        Returns:
            np.ndarray: array with n sample of a beta mixture
        """
        
        n_func = len(params_list)
        prop_sum = reduce(lambda x, y: x+y, [i[0] for i in params_list])
        if prop_sum != 1.0:
            print('Sum of lambdas is not equal 1')
        
        uniform_sample = np.random.uniform(0,1,size)
        
        if n_func == 2:
            beta_mixture = []
            
            for i in range(size):
                unif = random.uniform(0,1)
                if unif <= params_list[0][0]:
                    beta_mixture.append(random.betavariate(params_list[0][1], params_list[0][2]))
                elif unif > params_list[0][0]:
                    beta_mixture.append(random.betavariate(params_list[1][1], params_list[1][2]))

            self.__sample = np.array(beta_mixture)
            self.__x = np.linspace(0,1,size)

    
    def dist_freq(self):
        try:
            dens = gaussian_kde(self.__sample)
            y = dens(self.__x)
            self.__density = y
        except:
            print('Run beta_mix_sim() first')
    
    def normalize(self):
        total = self.__density.sum()
        norm = self.__density/total
        self.__normalized = norm
          
    @property
    def beta_mix(self):
        return self.__sample
            
    @property
    def density(self):
        return self.__density
            
    @property
    def x(self):
        return self.__x
    
    @property
    def normalized(self):
        return self.__normalized

          
        
class Interval:
    
    def __init__(self):
        self.__inf_lim = None
        self.__sup_lim = None
        self.__inf_index = None
        self.__sup_index = None

    def __get_inflim(self, vector, prop:float):
        for i in range(vector.shape[0]):
            acumu = vector[:i+1].sum()
            if acumu >= prop:
                return i+1
                
            else:
                continue
            
    def __get_supinf(self, vector, prop:float):
        for i in range(vector.shape[0]):
            acumu = vector[i+1:].sum()
            if acumu <= prop:
                return i+1
                
            else:
                continue
        
    def get_interval(self, y, interval:float, type:str = 'two_sided'):
        """ type: {less, two_sided, greater}"""
        if type == 'two_sided':
            area = (1-interval)/2
            inf_index = self.__get_inflim(y, area)
            sup_index = self.__get_supinf(y, area)
            inf_lim = y[inf_index]
            sup_lim = y[sup_index]
            
            self.__inf_lim = inf_lim
            self.__inf_index = inf_index
            self.__sup_lim = sup_lim
            self.__sup_index = sup_index
        
        elif type == 'less':
            area = (1-interval)
            inf_index = self.__get_inflim(y, area)
            inf_lim = y[inf_index]

            self.__inf_lim = inf_lim
            self.__inf_index = inf_index
        
        elif type == 'greater':
            area = (1-interval)
            sup_index = self.__get_supinf(y, area)
            sup_lim = y[sup_index]
            
            self.__sup_lim = sup_lim
            self.__sup_index = sup_index
        
        else:
            print('tipo errado')
            
    def plot_ginterval(self,x, y, type = 'two_sided'):
        if len(x) != len(y):
            print('the size of x and y must be the same.')
        
        if (type == 'two_sided') and (self.__inf_lim) and (self.__sup_lim):
            plt.figure()
            plt.plot(x, y, color='black')
            plt.fill_between(x, y, 0,\
                where= (x <= x[self.__inf_index]) | (x >= x[self.__sup_index]),\
                color = 'cadetblue')
            plt.show()
            
        elif (type == 'less') and (self.__inf_lim):
            plt.figure()
            plt.plot(x, y, color='black')
            plt.fill_between(x, y, 0,\
                where= (x <= x[self.__inf_index]),\
                color = 'cadetblue')
            plt.show()
        
        elif (type == 'greater') and (self.__sup_lim):
            plt.figure()
            plt.plot(x, y, color='black')
            plt.fill_between(x, y, 0,\
                where= (x >= x[self.__sup_index]),\
                color = 'cadetblue')
            plt.show()
        
        else:
            print('Run `get_interval` first')
    
    @property
    def inf_lim(self):
        return self.__inf_lim
    
    @property
    def sup_lim(self):
        return self.__sup_lim
    
    @property
    def inf_index(self):
        return self.__inf_index
    
    @property
    def sup_index(self):
        return self.__sup_index
