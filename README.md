# labs_workforce_forecasting

this depo contains code relevant to labs wf project and serves as an exchange buffer        

## how to run the code
              
1)download depo ZIP, unpack        
2)setup the environment              
in Anaconda Prompt or your local terminal       
update conda
```
conda update conda
```
locate binder folder 
```        
cd C:\MAIN\labs_workforce_forecasting-main\binder   
```         
create the env (may take some time)
```
conda env create -f environment.yml         
```     
(optional)update all packages
```
conda update --all    
```    
3)activate the env, run Jupyter Lab
locate main folder 
```        
cd C:\MAIN\labs_workforce_forecasting-main
```  
activate the environment
```        
conda activate simulation
```  
run JL
```        
jupyter lab
```
## how it works
this is a simulation model of a lab built on the basis of simpy package         
the arrivals generator mimics inflow of samples into the lab               
there are three types (bands) of employees ranging in their productivity and competencies              
all parameters are made-up and serve an illustratory purpose only           
the estimation of warm-up and number of runs are commented out for simplicity           

