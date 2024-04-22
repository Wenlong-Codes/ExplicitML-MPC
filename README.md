# Explicit Machine Learning-based Model Predictive Control of Nonlinear Processes via Multi-parametric Programming
## 1. Continuous Stirred Tank Reactor (CSTR) Example

- Let us consider a second-order, exothermic, irreversible reaction A >> B


<img src="https://github.com/Keerthana-Vellayappan/Demonstration-of-Physics-Informed-Machine-Learning-Model/assets/160836399/c1337cf1-eb78-47d7-b95b-1ce399d0ad10" alt = " Figure: Schematic diagram of a CSTR" width="250" height="250">


- The First Principle equations for this system are given as follows:


     <img src="![FP CSTR](https://github.com/Wenlong-Codes/ExplicitML-MPC/assets/140233129/2669a5c8-e05e-45a6-98ee-b5c7f9782a67)" width="300" height="90">


- Where,

   - 𝐶<sub>𝐴</sub>: Concentration of reactant A (kmol/m<sup>3</sup>)
   - 𝑇: Temperature of the reactor (K)
   - 𝐶<sub>𝐴0</sub>: Concentration of reactant A in the feed
   - 𝑄 :  Heat input rate (kJ/h)
   - F: feed volumetric flow rate (m<sup>3</sup>/h)
   - 𝑇<sub>0</sub>: Inlet temperature (K)

- The State and Manipulated variables for this system are:

    - States variables: _𝐱_=[𝐶<sub>A</sub>−𝐶<sub>As</sub>, 𝑇−𝑇<sub>s</sub>]
    - Control / Manipulated variables: _𝐮_=[𝐶<sub>A0</sub>−𝐶<sub>A0s</sub>, 𝑄−𝑄<sub>s</sub>]
