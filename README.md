# labs_workforce_forecasting

this depo contains code relevant to labs wf project and serves as an exchange buffer        

## how it works
this is a simulation model of a lab built on the basis of simpy package         
the arrivals generator mimics inflow of samples into the lab               
there are three types (bands) of employees ranging in their productivity and competencies      
lab processes samples at a given rate                
all parameters are made-up and serve an illustratory purpose only           
the estimation of warm-up and number of runs are removed for simplicity   
for further details please see the markdown cells in the code file           

## how to run the code
              
<b>1) download depo ZIP, unpack</b>        
<b>2) setup the environment</b>              
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
<b>3) activate the env, run Jupyter Lab</b>           
locate main folder 
```        
cd C:\MAIN\labs_workforce_forecasting-main
```  
activate the environment
```        
conda activate simulation
```  
```        
jupyter lab
```
