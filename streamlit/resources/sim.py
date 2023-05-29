#simulation model

import numpy as np
import pandas as pd
import itertools
import math
import matplotlib.pyplot as plt
import simpy
from joblib import Parallel, delayed
import warnings
from scipy.stats import t
#from treat_sim.distributions import Exponential, Lognormal


# Distribution classes


class Exponential:
    '''
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, mean, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the exponential distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.exponential(self.mean, size=size)

class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """
    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the lognormal distribution
        stdev = standard dev of the lognormal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        
    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m = mean of lognormal distribution
        v = variance of lognormal distribution
                
        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma
        
    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma)


# Utility functions


def trace(msg):
    '''
    Utility function for printing simulation
    set the TRACE constant to FALSE to 
    turn tracing off.
    
    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if TRACE:
        print(msg)


# Model parameters

# These are the parameters for a base case model run.

# run length in days
RUN_LENGTH = 365

# audit interval in days
DEFAULT_WARMUP_AUDIT_INTERVAL = 1

# default № of reps for multiple reps run
DEFAULT_N_REPS = 51

# default random number SET
DEFAULT_RNG_SET = None
N_STREAMS = 10

# Turn off tracing
TRACE = False

# resource counts
N_BEDS = 9

# time between arrivals in minutes (exponential)
# for acute stroke, TIA and neuro respectively
MEAN_IATs = [1.2, 9.5, 3.5]

# treatment (lognormal)
# for acute stroke, TIA and neuro respectively
TREAT_MEANs = [7.4, 1.8, 2.0]
TREAT_STDs = [8.5, 2.3, 2.5]


#Scenario class
class Scenario:
    '''
    Parameter container class for ASU model.
    '''

    def __init__(self, random_number_set=DEFAULT_RNG_SET):
        '''
        Initialize the Scenario object with default values.

        Parameters:
        ----------
        random_number_set: int, optional
            The random number set to be used by the simulation.
        '''

        # Warm-up period
        self.warm_up = 0.0

        # Default values for inter-arrival and treatment times
        self.iat_means = MEAN_IATs
        self.treat_means = TREAT_MEANs
        self.treat_stds = TREAT_STDs

        # Sampling
        self.random_number_set = random_number_set
        self.init_sampling()

        # Number of beds
        self.n_beds = N_BEDS

    def set_random_no_set(self, random_number_set):
        '''
        Set the random number set to be used by the simulation.

        Parameters:
        ----------
        random_number_set: int
            The random number set to be used by the simulation.
        '''
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        '''
        Initialize the random number streams and create the distributions used by the simulation.
        '''

        # Create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)

        # Initialize the random seeds for each stream
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)

        # Create inter-arrival time distributions for each patient type
        self.arrival_dist_samples = {
            'stroke': Exponential(self.iat_means[0], random_seed=self.seeds[0]),
            'tia': Exponential(self.iat_means[1], random_seed=self.seeds[1]),
            'neuro': Exponential(self.iat_means[2], random_seed=self.seeds[2])
        }

        # Create treatment time distributions for each patient type
        self.treatment_dist_samples = {
            'stroke': Lognormal(self.treat_means[0], self.treat_stds[0], 
                                random_seed=self.seeds[3]),
            'tia': Lognormal(self.treat_means[1], self.treat_stds[1], 
                             random_seed=self.seeds[4]),
            'neuro': Lognormal(self.treat_means[2], self.treat_stds[2], 
                               random_seed=self.seeds[5])
        }

        
# Model building

class Patient:
    '''
    Patient in the ASU processes
    '''
    def __init__(self, identifier, patient_type, env, args):
        '''
        Constructor method
        
        Params:
        -----
        identifier: int
            a numeric identifier for the patient.
            
        env: simpy.Environment
            the simulation environment
            
        args: Scenario
            The input data for the scenario
        '''
        # patient id and environment
        self.identifier = identifier
        self.env = env
        
        # treatment parameters
        self.patient_type = patient_type
        self.beds = args.beds
        self.treatment_dist_samples = args.treatment_dist_samples
                
        # individual patient metrics
        self.queue_time = 0.0
        self.treat_time = 0.0
    
    def get_treatment_dist_sample(self):
        '''
        This method returns a sample from the treatment distribution of the patient, based on their type.
        '''
        self.treat_time = self.treatment_dist_samples[self.patient_type].sample()
        return self.treat_time
    
    def treatment(self):
        '''
        This method represents the patient's treatment process. The patient will request a bed, wait in the queue,
        and then undergo treatment before being discharged.
        '''
        # record the time that patient entered the system
        arrival_time = self.env.now
     
        # get a bed
        with self.beds.request() as req:
            yield req
            
            # calculate queue time and log it
            self.queue_time = self.env.now - arrival_time
            trace(f'Patient № {self.identifier} started treatment at {self.env.now:.3f};' 
                 + f' queue time was {self.queue_time:.3f}') 
            
            # wait for treatment to finish
            yield self.env.timeout(self.get_treatment_dist_sample())
            
            # discharge the patient
            self.patient_discharged()
    
    def patient_discharged(self):
        '''
        This method logs the patient's discharge and frees up the bed.
        '''
        trace(f'Patient № {self.identifier} discharged at {self.env.now:.3f}')
class MonitoredPatient(Patient):
    '''
    A MonitoredPatient class which monitors a patient process and notifies its observers 
    when a patient process has reached an event of completing treatment.
    
    This class implements the observer design pattern.
    '''
    
    def __init__(self, admissions_count, patient_type, env, args, model):
        '''
        Constructor for MonitoredPatient class.
        
        Params:
        -------
        admissions_count: int
            The identifier for the patient
            
        patient_type: str
            The type of patient, either 'stroke', 'tia', or 'neuro'
            
        env: simpy.Environment
            The simulation environment
            
        args: Scenario
            The input data for the scenario
            
        model: Model
            The model to be observed
        '''
        
        # Calls the constructor for the Patient superclass
        super().__init__(admissions_count, patient_type, env, args)
        
        # Creates a list of observers to notify
        self._observers = [model]
        
    def register_observer(self, observer):
        '''
        A method to register an observer to be notified when an event occurs.
        
        Params:
        -------
        observer: Observer
            The observer to be registered
        '''
        
        # Adds the observer to the list of observers
        self._observers.append(observer)
    
    def notify_observers(self, *args, **kwargs):
        '''
        A method to notify all registered observers when an event occurs.
        
        Params:
        -------
        *args: Any
            Positional arguments passed to the observer method
        
        **kwargs: Any
            Keyword arguments passed to the observer method
        '''
        
        # Calls the process_event method on each observer with the arguments passed
        for observer in self._observers: 
            observer.process_event(*args, **kwargs)
    
    def patient_discharged(self):
        '''
        A method to notify all observers that the patient has been discharged.
        '''
        
        # Calls the patient_discharged method on the Patient superclass
        super().patient_discharged()
        
        # Notifies all observers that the patient has been discharged
        self.notify_observers(self, 'patient_discharged')
class ASU:  
    '''
    Model of an ASU
    '''
    def __init__(self, args):
        '''
        Contructor
        
        Params:
        -------
        env: simpy.Environment
        
        args: Scenario
            container class for simulation model inputs.
        '''
        self.env = simpy.Environment()
        self.args = args 
        self.init_model_resources()
        self.patients = []
        
        self.arrivals_count = 0
        
        self.stroke_count = 0
        self.tia_count = 0
        self.neuro_count = 0
        
        #running performance metrics:
        self.bed_wait = 0.0
        self.bed_util = 0.0
        
        self.patient_count = 0
            
        self.bed_occupation_time = 0.0
        
        
    def init_model_resources(self):
        '''
        Setup the simpy resource objects
        
        Params:
        ------
        args - Scenario
            Simulation Parameter Container
        '''

        self.args.beds = simpy.Resource(self.env, 
                                   capacity=self.args.n_beds)
        
        
    def run(self, results_collection_period = RUN_LENGTH,
            warm_up = 0):
        '''
        Conduct a single run of the model in its current 
        configuration

        run length = results_collection_period + warm_up

        Parameters:
        ----------
        results_collection_period, float, optional
            default = RUN_LENGTH

        warm_up, float, optional (default=0)
            length of initial transient period to truncate
            from results.

        Returns:
        --------
            None

        '''
        
        # setup the arrival processes
        self.env.process(self.arrivals_generator('stroke'))
        self.env.process(self.arrivals_generator('tia'))
        self.env.process(self.arrivals_generator('neuro'))
                
        # run
        self.env.run(until=results_collection_period+warm_up)
        
        
    def get_arrival_dist_sample(self):
        
        inter_arrival_time = self.args.arrival_dist_samples[self.patient_type].sample()
        return inter_arrival_time
            
        
    def arrivals_generator(self, patient_type):
        self.args.init_sampling()
            
        while True:
                
            self.patient_type = patient_type    

            iat = self.get_arrival_dist_sample()
            yield self.env.timeout(iat)
                
            if self.env.now > self.args.warm_up:    
                self.arrivals_count += 1

            trace(f'Patient № {self.arrivals_count} ({patient_type}) arrives at {self.env.now:.3f}')
                
            new_patient = MonitoredPatient(self.arrivals_count, patient_type, self.env, self.args, self)                

            self.env.process(new_patient.treatment())                 
                               
    
    
    def process_event(self, *args, **kwargs):
        '''
        Running calculates each time a Patient process ends
        (when a patient departs the simulation model)
        
        Params:
        --------
        *args: list
            variable number of arguments. This is useful in case you need to
            pass different information for different events
        
        *kwargs: dict
            keyword arguments.  Same as args, but you can is a dict so you can
            use keyword to identify arguments.
        
        '''
        patient = args[0]
        msg = args[1]
        
        #only run if warm up complete
        if self.env.now < self.args.warm_up:
            return

        if msg == 'patient_discharged':
            
            self.patients.append(patient)
            
            if self.patient_type == 'stroke':
                self.stroke_count += 1
            elif self.patient_type == 'tia':
                self.tia_count += 1
            else:
                self.neuro_count += 1
                
            self.patient_count += 1
            n = self.patient_count
            
            #running calculation for mean bed waiting time
            self.bed_wait += \
                (patient.queue_time - self.bed_wait) / n

            #running calc for mean bed utilisation
            self.bed_occupation_time += patient.treat_time

                
                
    def run_summary_frame(self):
        
        '''
        Utility function for final metrics calculation.

        Returns a pandas DataFrame containing summary statistics of the simulation.
        '''
        
        # adjust util calculations for warmup period
        rc_period = self.env.now - self.args.warm_up
        util = self.bed_occupation_time / (rc_period * self.args.n_beds)
        
        # create nparray of all queue times, convert to hours
        patients_queue_times = np.array([patient.queue_time * 24 for patient in self.patients])
        
        # Find the value at the 90th percentile
        pct_90 = np.percentile(patients_queue_times, 90)

        # Filter out any values above the 90th percentile
        filtered_times = patients_queue_times[patients_queue_times <= pct_90]

        # Calculate the mean of the filtered times
        bed_wait_90 = np.mean(filtered_times) 
        

        # calculate proportion of patient with queue time less than 4 hrs
        percent_4_less = (sum(qt <= 4 for qt in patients_queue_times) / len(self.patients)) * 100
        
        bed_wait = self.bed_wait * 24


        df = pd.DataFrame({'1':{'0 Total Patient Arrivals':self.arrivals_count,
                                '1a Total Patient Admissions':self.patient_count,
                                '1b Stroke Patient Admissions':self.stroke_count,
                                '1c TIA Patient Admissions':self.tia_count,
                                '1d Neuro Patient Admissions':self.neuro_count,
                                '2 Mean Queue Time (hrs)':bed_wait,
                                '3 Mean Queue Time of Bottom 90% (hrs)': bed_wait_90,
                                '4 Patients Admitted within 4 hrs of arrival(%)': percent_4_less,
                                '5 Bed Utilisation (%)': util*100}})

                                
        df = df.T
        df.index.name = 'rep'
        return df

    
# Functions for single and multiple runs
def single_run(scenario, 
               rc_period = RUN_LENGTH, 
               warm_up = 0,
               random_no_set = DEFAULT_RNG_SET):
    '''
    Perform a single run of the model and return the results
    
    Parameters:
    -----------
    
    scenario: Scenario object
        The scenario/paramaters to run
        
    rc_period: int
        The length of the simulation run that collects results
        
    warm_up: int, optional (default=0)
        warm-up period in the model.  The model will not collect any results
        before the warm-up period is reached.  
        
    random_no_set: int or None, optional (default=1)
        Controls the set of random seeds used by the stochastic parts of the 
        model.  Set to different ints to get different results.  Set to None
        for a random set of seeds.
        
    Returns:
    --------
        pandas.DataFrame:
        results from single run.
    '''  
        
    # set random number set - this controls sampling for the run.
    if random_no_set is not None:
        scenario.set_random_no_set(random_no_set)
    
    scenario.warm_up = warm_up
    
    # create the model
    model = ASU(scenario)

    model.run(results_collection_period = rc_period, warm_up = warm_up)
    
    # run the model
    results_summary= model.run_summary_frame()
    
    return results_summary

def multiple_replications(scenario, 
                          rc_period=RUN_LENGTH,
                          warm_up=0,
                          n_reps=DEFAULT_N_REPS, 
                          n_jobs=-1):
    '''
    Perform multiple replications of the model.
    
    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model
    
    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model beyond warm up
        to collect results
    
    warm_up: float, optional (default=0)
        initial transient period.  no results are collected in this period

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.
        
        
    Returns:
    --------
    List
    '''    
    
    random_no_set = scenario.random_number_set
    
    if random_no_set is not None:
        rng_sets = [random_no_set + rep for rep in range(n_reps)]
    else:
        rng_sets = [None] * n_reps
       
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(scenario, 
                                                      rc_period, 
                                                      warm_up, 
                                                      random_no_set=rng_set) 
                                    for rng_set in rng_sets)
    

    # format and return results in a dataframe
    df_results = pd.concat(res)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results

