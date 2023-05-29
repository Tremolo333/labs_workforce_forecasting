from resources.sim import Scenario, multiple_replications
import streamlit as st
import glob
import os

INTRO_FILE = 'resources/overview.md'

def read_file_contents(file_name):
    ''''
    Read the contents of a file.

    Params:
    ------
    file_name: str
        Path to file.

    Returns:
    -------
    str
    '''
    with open(file_name) as f:
        return f.read()

# give the page a title
st.title('Acute Stroke Unit (ASU) Simulation Model')

# show the introductory markdown
st.markdown(read_file_contents(INTRO_FILE))

# Using "with" notation
with st.sidebar:

    # set the variables for the run.
    # these are just a subset of the total available for this example...
    # in streamlit we are going to set these using sliders.

    # Reproducible run switch
    reproducible_run = st.checkbox('Reproducible run',
                            help = 'Check for controlled sampling')
    if reproducible_run:
        rng_set = st.slider("Select seed value", 0, 999, 333)
    
     # Number of runs
    replications = st.slider('No. replications', 1, 100, 51,
                            help = 'Max 50 rep is supported')

    # Number of beds
    n_beds = st.slider('Number of beds', 1, 30, 9, 1)

    #Interarrival time
    stroke_iat = st.slider('Interarrival time for stroke patient', 
                           1.0, 5.0, 1.2, 0.01,
                          help = 'Patients who have suffered an acute stroke')
    tia_iat = st.slider('Interarrival time for TIA patient',
                        1.0, 15.0, 9.5, 0.01,
                       help = 'Patients who have suffered a transient ischaemic attack (TIA)')
    neuro_iat = st.slider('Interarrival time for neuro patient', 
                          1.0, 10.0, 3.5, 0.01,
                         help = 'Patients who have complex neurological conditions')
    
    # Treatment time mean
    stroke_treat_mean = st.slider('Mean length of stay for stroke patient', 
                                  1.0, 20.0, 7.4, 0.1,
                                 help = 'Patients who have suffered an acute stroke')
    tia_treat_mean = st.slider('Mean length of stay for patient for TIA patient',
                               1.0, 20.0, 1.8, 0.1,
                              help = 'Patients who have suffered a transient ischaemic attack (TIA)')
    neuro_treat_mean = st.slider('Mean length of stay for patient for neuro patient', 
                                 1.0, 20.0, 2.0, 0.1,
                                help = 'Patients who have complex neurological conditions')
    
    # Treatment time std
    stroke_treat_std = st.slider('SD of length of stay for stroke patient', 
                                 1.0, 20.0, 8.5, 0.1,
                                help = 'Patients who have suffered an acute stroke')
    tia_treat_std = st.slider('SD of length of stay for patient for TIA patient', 
                              1.0, 20.0, 2.3, 0.1,
                             help = 'Patients who have suffered a transient ischaemic attack (TIA)')
    neuro_treat_std = st.slider('SD of length of stay for patient for neuro patient', 
                                1.0, 20.0, 2.5, 0.1,
                               help = 'Patients who have complex neurological conditions')
    


# Setup scenario using supplied variables
args = Scenario()

if reproducible_run:
    args.random_number_set = rng_set

args.n_beds = n_beds

args.iat_means = [stroke_iat, tia_iat, neuro_iat] 
args.treat_means = [stroke_treat_mean, tia_treat_mean, neuro_treat_mean]
args.treat_stds = [stroke_treat_std, tia_treat_std, neuro_treat_std]

if st.button('Simulate ASU'):
    # in this example run a single replication of the model.
    with st.spinner('Simulating the ASU...'):
        print(replications)
        results = multiple_replications(args, n_reps=replications, warm_up = 250, )

    st.success('Done!')

    st.table(results.mean().round(2))
