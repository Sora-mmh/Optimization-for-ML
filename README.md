# Optimization-for-ML

This project is about different optimization algorithms for machine learning in different settings (constrained, unconstrained, automatic differentiation...).

A quick overview which is more detailed in this [experimentations and observations notebook](Optimization-for-ML/experimentations_and_observations.ipynb) :

- [Part 1: Exploratory Data Analysis about Diabetes dataset](#1)
  - [1.1 Import Dataset](#1.1)
  - [1.2 Exploration](#1.2)
- [Part 2: Gradient Descent](#2)
  - [2.1 Logistic Regression GD for classification](#2.1)
  - [2.2 Step Size](#2.2)
  - [2.3 Convergence Rate](#2.2)
  - [2.4 Performance Test Set (change the RIDGE penalty)](#2.4)
- [Part 3: Automatic Differentiation](#3)
- [Part 4: Stochtastic Gradient Descent](#4)
  - [4.1 SGD and Comparison with GD](#4.1)
  - [4.2 Optimal Batch Size](#4.2)
  - [4.3 SVRG](#4.3)
- [Part 5: Convexity and Constrained Optimization](#5)
  - [5.1 Conditional Gradient Algorithm](#5.1)
  - [5.2 Projected Gradient Optimization](#5.2)
- [Part 6: Regularization](#6)
  - [6.1 L-1 Regularization : LASSO](#6.1)
  - [6.2 L-2 Regularization](#6.2)
- [Part 7: Large-Scale and Distributed Optimization](#7)
  - [7.1 Randomized Block Coordinate Descent](#7.1)
  - [7.2 Randomized Block Coordinate Descent + SGD](#7.2)
- [Part 8: Advanced Topics On Gradient Descent](#8)
  - [8.1 Heavy Ball](#8.1)
  - [8.2 Add Non-Convex Penalty to the loss (L-0.5 penalty) + GD](#8.2)
