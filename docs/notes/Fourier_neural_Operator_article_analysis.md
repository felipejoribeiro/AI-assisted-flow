## Analysis on the article **Fourier Neural Operator For Parametric Partial Differential Equations**.
Here it will be discussed the information provided by the scientific article about neural network assisted computational fluid dinamics.

Some other works from the authors:
- Neural operator graph kernel network for partial differential equations.
- Kernel methods for deep learning.

### Abstract
- Classical neural networks focused on learning mappings between finite-dimensional euclidean spaces.
- This was generalized to neural operators that learn mappings between function spaces.
- Any mapping of functional parametric dependence to the solution on PDEs can be learned by neural operators.
    - This enable then to learn an entire family of PDEs.
    - In classical methods they could learn only one instance of the equation.
	- About kernel methods: Kernel methods for deep learning.pdf
- In this work, we formulate a new neural operator by parametrized the integral kernel directly in fourier space. That is, developing the learning process in it.
- The result was an expressive and efficient architecture.
- Experimentations were performed in Burgers' equation, Darcy flow and Navier-Stokes equation.
- The Fourier neural operator (the one developed in this work), is the first ML-based method to successfully model turbulent flows with zero-shot super-resolution.
    - zero-shot super-resolution ?

### Introduction
Manu problems consist in solving complex partial differential equations (PDE). Traditional discrete approaches are costly and not viable in many occasion given the complexity of the problem.

- Traditional solvers such as finite element methods (FEM) and finite difference methods (FDM) sove by discretization of space. There is a trade off between accuracy and computational cost linked to discretization size.

- Data-driven methods can learn the trajectory of the family of equations from the data. It can be orders of magnitude faster that traditional methods. But there is a catch, traditional neural networks can only learn solutions tied to a specific discretization. This is a limitation for practical applications.

- The development of mesh-invariant neural networks is required. There are two mainstream neural network-based approaches for PDEs - the finite-dimensional operators and Neural-FEM:
	- Finite-dimensional operators: This approach parametrize the solution operator as a deep convolucional neural network between finite-dimensional euclidean spaces. These are, by definition mesh-dependent and geometry-dependent on training data. Some works that can be read to better anderstand the approach:
		- Prediction of aerodynamic flow fields using convolucional neural networks.
		- Solving_PDE_problems_with_uncertainty_using_neural-networks.
		- Convolucional neural networks for steady flow approximation.
	- Neural-FEM:


