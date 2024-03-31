Download Train1.mat and Train2.mat from the Resources section of Piazza, two sets of training data each with n = 100 and p = 2. Using the formulas from the class notes, code a (full, i.e., not stochastic) gradient descent algorithm for learning the decision hyperplane parameters w and b for the soft margin SVM. Your implementation should work for arbitrary n and p. Turn a documented listing of your code. 

(a) Initialize w and b to random values and plot the resulting (random) hyperplane (a straight line since p = 2) on top of the scatter plot of the Train1.mat data. 

(b) Set up your gradient descent code such that it redraws the decision hyperplane each iteration of training. Run your code to convergence on Train1.mat, showing the progression of hyperplanes for λ = 1. Discuss the convergence, in particular to the ultimate support vectors. Try with a few different initializations for w and b. 

(c) Re-run the experiment from (c) with various values of λ and comment on the results. In particular, what does λ trade off? 

(d) Modify your code to take stochastic gradient steps rather than full gradient steps. What changes? 

(e) Now use the same implementation to fit an SVM to the Train2.mat data with λ = 1. Plot the learned decision hyperplane on top of the scatter plot of the data and note the misclassification error. Would you call this a good classifier? Why or why not? 

(f) Experiment with various nonlinear functions ϕ that map the data points from xi ∈ Rp=2 to ϕ(xi) ∈ RP=3. Plot the remapped data using a 3D scatter plot to visualize whether your nonlinear function is successful in making the data near linearly separable.ELEC 378– Spring 2024– Homework 6 2 

(g) When you have found an appropriate transform ϕ, use the same implementation to fit an SVM to the Train2.mat data after transforming each data point from xi to ϕ(xi). How does the resulting misclassification error compare to your SVM fit without transforming the data? Optional (but highly recommended): Plot the optimal hyperplane overlaid on the 3D scatter plot.