# Stochastic Programming Case Study 8.1
Repository for the case study 8.1 [Haneveld at al., Section 8.1] for the [LNMB Stochastic Programming](https://www.lnmb.nl/pages/courses/phdcourses/SP.html) 2022 PhD course.

---
## Implementation
### Structure 
The repository is structured in the following way
- `results`: this folder contains the pickle- and json-serialized results of each simulations carried to the end. These files shall contain both the arguments used to generate them and the resulting numerical outputs of the various optimization procedures.
- `src`: this folder contains the actual Python files used in solving the case study problems
    - `main.py`: as the name suggests, this is the entry point of the code. In here, we simply parse the command line arguments, build the various constant parameters of the problem, and then run the actual methods for solving the multiple points of the case study. These are divided in *recourse* (points a, b, and c in the textbook) and *chance constraint* model-related (point d). Finally, the last step is to collect the results from these points and save them.
    - `util.py`: contains all the utility functions for, e.g., parsing arguments, generating the problem constant parameters, drawing samples via LHS, saving results, etc..
    - `recourse.py`: contains all the methods (called by `main.py`) to solve the *recourse* model-related tasks. In order, they consist of
        1. computing the Expected Value (EV)
        2. computing the Expected Result of the EV (EEV)
        3. solving the Two-Stage model (TS) as a Large Scale Deterministic Equivalent (LSDE)
        4. assessing the quality of the TS solution via Multiple Replications Procedure (MRP)
        5. computing the Wait-and-See solution (WS)
        6. computing the Value of the Stochastic Solution (VSS) and the Expected Value of Perfect Information (EVPI)
        7. computing the sensitivity analysis of the TS solution to the parameters concerning the labor extra capacity increase upper bound and associated cost
        8. computing the sensitivity analysis of the TS solution to the demand distributions dependence, both inter-product and inter-period
    - `recourse.py`: contains all the methods (called by `main.py`) to solve the *chance constraint* model-related tasks. <span style="color:red">TODO</span>


### How to run 
The code can be run via the simple command:
```bash
python main.py
```
As aforementioned, there is plethora of additional arguments that can be passed
to the script. They are the following
- **-iv, --intvars**: a flag that tells the Gurobi models to use integer variables instead of continuous. If not specified, by default continuous variables are employed.
- **-s, --samples**: the number of samples to be drawn in order to approximate the continuous product demand distributions as discrete distributions. This sample size will be used in most of the computations, so it has a huge impact on computation time. By default, the size is 10000.
- **-a, --alpha**: confidence level for the MRP, which will output a confidence interval scaled according to the user-specified $\alpha$. By default, $\alpha$ is 0.95.
- **-r, --replicas**: number of replications for the MRP, which will output a confidence interval based on average and standard deviation computed across this many repliactions. By default, the replications are 30.
- **-lf, --lab_factors**: a list of factors for the sensitivity analysis of labor. These factors modify two of the problem parameters, namely the labor extra capacity upper bound and associated cost. Assuming $N$ factors are specified, then the whole $N^2$ combinations of the two factors are generated, and for each a new TS solution is computed. The list of factors has to be specified as a string, and the default factors are "[.8, 1, 1.2]".
- **-df, --dep_factors**: a list of factors for the sensitivity analysis of demand distributions dependency. In the original TS model, these ditributions are assumed to be independent between different products and at different time steps. To put this independence assumption to the test, these factors will be used to create multivariate normal distributions with some covariance between different products in the same period, and covariance of the same product at different periods. No correlation between different products in different periods is tested. Assuming $N$ factors are specified, then the whole $N^2$ combinations of the two factors are generated, and for each new samples are drawn and a new TS solution is computed. The list of factors has to be specified as a string, and the default factors are "[-0.1, 0, 0.1]".
- **-s, --seed**: random number generator seed. By default, the seed is None.
- **-v, --verbose**: verbosity level of the Gurobi solvers. It can be 0, 1 or 2, where 0 is the less verbose. By default, the verbosity level is 0.

A complete example would be
```bash
python main.py --intvars --samples 10000 --alpha 0.95 --replicas 30 --lab_factors "[0.8, 0.9, 1.0, 1.1, 1.2]" --dep_factors "[0.1, 0.2, 0.3, 0.4, 0.5]" --seed 42 --verbose 0
```

---
## Authors
Alban Kryeziu (a.kryeziu@rug.nl) and Filippo Airaldi (f.airaldi@tudelft.nl)

---
## References
Haneveld, Willem K. Klein, Maarten H. Van der Vlerk, and Ward Romeijnders. *Stochastic Programming: Modeling Decision Problems Under Uncertainty*. Springer Nature, 2019.
