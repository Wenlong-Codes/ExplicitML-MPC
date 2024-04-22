# Explicit Machine Learning-based Model Predictive Control of Nonlinear Processes via Multi-parametric Programming
## 1. Continuous Stirred Tank Reactor (CSTR) Example

- Let us consider a second-order irreversible exothermic reaction A >> B:


<img src="https://github.com/Keerthana-Vellayappan/Demonstration-of-Physics-Informed-Machine-Learning-Model/assets/160836399/c1337cf1-eb78-47d7-b95b-1ce399d0ad10" alt = " Figure: Schematic diagram of a CSTR" width="250" height="250">


- The first-principles equations (i.e., energy and material balance equations) for this system are given as follows:


     [<img src="assets/FP CSTR.jpg">](https://github.com/Wenlong-Codes/ExplicitML-MPC/)


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

## 2. Develop Explicit ML-MPC for the CSTR

- Step 1: Perform extensive open-loop simulations to obtain sufficient data for model training.
     > Please refer to the files under the "Open-loop Simulation" folder
- Step 2: Build a ML model to capture the nonlinear system dynamics of the CSTR.
     > Please refer to the files under the "Train_Model" folder
- Step 3: Approximate the nonlinear behavior of the trained ML model using piecewise linear affine functions.
     > Please refer to the files under the "Space_discretization" folder
- Step 4: Generate the solution maps for the discretized state-space via multi-parametric programming.
     > Please refer to the file "GetExplicitML-MPC_Sols.py"
- Step 5: Carry out close-loop simulations to test the effectiveness of the proposed Explicit ML-MPC.
     > Please refer to the file "ExplicitML-MPC.py"
  
## 3. Citation
You can find our paper [here](https://www.sciencedirect.com/science/article/abs/pii/S0098135424001078)
> @article{wang2024explicit,
  title={Explicit machine learning-based model predictive control of nonlinear processes via multi-parametric programming},
  author={Wang, Wenlong and Wang, Yujia and Tian, Yuhe and Wu, Zhe},
  journal={Computers \& Chemical Engineering},
  pages={108689},
  year={2024},
  publisher={Elsevier}
}
