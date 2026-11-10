import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

from NeuralNetworkClass import NeuralNetwork
from NeuralNetworkWrapper import NeuralNetRegressor
from Functions import ReLU, ReLU_der, sigmoid, sigmoid_der, mse, mse_der, identity, identity_der, leaky_ReLU, leaky_ReLU_der

#---------------------------------------------------------------------

# set theme for plots, grid style, and font sizes 
sns.set_theme(style="whitegrid",
    rc={ "axes.titlesize": 18,   
         "axes.labelsize": 20,  
         "xtick.labelsize": 18, 
         "ytick.labelsize": 18, 
         "legend.fontsize": 16,
         "legend.title_fontsize": 10})

#---------------------------------------------------------------------

def runge_function(x):
    return 1 / (1 + 25 * x**2)

""" Generates input & output data from a function given as parameter, with zero-mean Gaussian noise. """
def generate_data(f, n=100, sigma=0.1, seed=2026):

    rng = np.random.default_rng(seed) # random number generator with fixed seed
    x = np.sort(rng.uniform(-1, 1, n)).reshape(-1, 1) # generate input values
    epsilon = rng.normal(0, sigma, x.shape) # additive gaussian noise
    y = f(x) + epsilon # output data
    return x, y

#---------------------------------------------------------------------

# Generate data with 100 data points and noise variance 0.01
x_, y_  = generate_data(runge_function)
# Split data into training (80%) and test data (20%)
x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=2026)

# Variance of training observations (the MSE of predicting the mean value)
var_y = np.var(y_train)
print("-"*80)
print(f"Variance of training observations: var(y) = {var_y:.3f}")
print("-"*80)

# Setup 5-fold cross-validation with shuffle and fixed seed used througout the analysis
cv = KFold(n_splits=5, shuffle=True, random_state=2026)

#---------------------------------------------------------------------

def within_one_standard_error(mse, se, D2=False):
    """ Find cross-validation MSE and SE within one standard error of the minimum MSE and corresponding indices. """

    # Find index with lowest mse
    best_index = np.unravel_index(np.argmin(mse), mse.shape)
    # value of best mse with standard error
    best_mse_with_se = mse[best_index] + se[best_index]
    # find all candidates that are within one standard error
    candidate = mse <= best_mse_with_se 

    if D2: # if mse is 2 dimentional
        row, column  = np.where(mse <= best_mse_with_se)
        return candidate, row, column
    else:
        index = np.where(mse <= best_mse_with_se)
        return candidate, index

#---------------------------------------------------------------------

def polynomial_CV(x_train, y_train, method, degrees, lmbd=[]):
    """ Estimate cross-validation MSE, SE and mean time fit time for regression methods OLS, Ridge and Lasso. """
    # make pipeline with polynomial features standardization and regression method:
    if method=="OLS": # Select OLS
        poly = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), LinearRegression())
        degree_lam = {"polynomialfeatures__degree": degrees} # dictionary key and list of parameters (gridsearch only on degree)
        i, j = len(degrees), 1 # (degree, 1)

    elif method=="Ridge": # Select Ridge
        poly = make_pipeline( PolynomialFeatures(include_bias=False), StandardScaler(), Ridge())
        degree_lam =  {"polynomialfeatures__degree": degrees, "ridge__alpha": len(x_train) * lmbd} # dictionary key and list of parameters  
        i, j = len(degrees), len(lmbd) # (degree, lambda)

    elif method=="Lasso": # Select Lasso
        poly = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), Lasso(max_iter=100000, tol=1e-5))
        degree_lam =  {"polynomialfeatures__degree": degrees, "lasso__alpha": lmbd / 2} # dictionary key and list of parameters  
        i, j =  len(lmbd), len(degrees) # (lambda, degree)
    else:
        raise ValueError("Valid method names: OLS, Ridge, Lasso")

    # performe grid search on polynomial pipeline
    search_poly = GridSearchCV(poly, degree_lam, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)
    search_poly.fit(x_train, y_train) # fit training data
    results_poly = search_poly.cv_results_ # get results 
    mse = -results_poly["mean_test_score"].reshape(i, j) # CV MSE
    se = results_poly["std_test_score"].reshape(i, j) / np.sqrt(cv.get_n_splits() - 1)  # CV SE: std_test_score uses ddof=0: s/sqrt(k) = std/sqrt(k-1)
    time_fit = results_poly["mean_fit_time"].reshape(i, j) # mean fit time [s]

    # print(results_poly["params"]) # print parameters to check shapes

    if method=="OLS":
        return mse.ravel(), se.ravel(), time_fit.ravel()
    else:
        return mse, se, time_fit    

#---------------------------------------------------------------------

# #---------------------------------------------------------------------
# OLS CV mse and se
# #---------------------------------------------------------------------
mse_OLS, se_OLS, time_fit_OLS = polynomial_CV(x_train, y_train, method="OLS", degrees=range(1, 16))
candidate_OLS, index_OLS = within_one_standard_error(mse_OLS, se_OLS)

print("OLS: Polynomial degrees that are within one standard error")
for i in index_OLS[0]:
    print(f"Polynomial degree: {i + 1}, CV MSE: {mse_OLS[i]:>5.3f} pm {se_OLS[i]:>5.3f}, mean time to fit: {time_fit_OLS[i]:>5.6f}")
print("-"*80)

#---------------------------------------------------------------------


# ------------------------
# # Part b)
# ------------------------


"""
LLM-assisted
------------
Tool: OpenAI Codex (GPT-6 Astra) (Novermber 2026)
Role: generated plotting code for 1x2 plot with additonal code for correcting placement of legend and lables.
Modifications: adapted the plotting code to our existing loops.
"""

def plot_1x2_MSE_epochs_(penalty_method="L2", MSE=0.027, SE=0.007, parameters_50=[], parameters_100_100=[], optimizer_method="ADAM"): 
    """ 1x2 plot of cross-validation MSE and SE as function of epochs, for two different configurations ([50], [100, 100])"""
    """
    parameters_50: list of chosen parameters for hidden layer [50] 
    parameters_100_100 : list of chosen parameters for hidden layers [100, 100]
              [penalty, learning rate, batch size]
    """
    # get penalty values to set correct colors and lables.
    lam_50 = parameters_50[0]
    lam_100 = parameters_100_100[0]
    # get correct naming dependent on method
    if penalty_method=="L2" and (lam_50>0.0 or lam_100>0.0):
        color_ = "green"
        lable_ = "Ridge"
    elif penalty_method=="L1" and (lam_50>0.0 or lam_100>0.0):
        color_ = "blue"
        lable_ = "Lasso"
    else:
        color_ = "red"
        lable_ = "OLS"

    hidden_layer_lables = ["[50]", "[100, 100]"]

    # interval of epochs
    epochs = [50, 100, 300, 1000, 2000, 3000]

    # list of parameters for hidden layers [50] and [100, 100]
    lambdas = [parameters_50[0], parameters_100_100[0]]     # penalty
    gammas = [parameters_50[1], parameters_100_100[1]]      # learning rate
    batch_sizes= [parameters_50[2], parameters_100_100[2]]  # batch size

    layer_output_sizes_list = [[50, 1],
                            [100, 100, 1]]

    activation_funcs_list = [[sigmoid, identity],
                            [sigmoid, sigmoid, identity]]

    activation_ders_list = [[sigmoid_der, identity_der],
                            [sigmoid_der, sigmoid_der, identity_der]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, layout="constrained")
    # loops over configuration with corresponing sizes activation functions, derivative of activation functions, learning rate, penalty and batch size.
    for ax, layer_output_size, activation_func, activation_der, hidden_layer_lable, gamma, lam, M in zip(
        axes.flat, layer_output_sizes_list, activation_funcs_list, activation_ders_list, hidden_layer_lables, gammas, lambdas, batch_sizes):

        """Cheack if implementation of loop output and function inputs have correct values""" 
        # print(f"layer {layer_output_size}") 
        # print(f"Activation functions: {[func.__name__ for func in activation_func]}") # loop over activation functions and get the names of the function
        # print(f"Activation derivative: {[der.__name__ for  der in activation_der]}") # loop over activation derivative and get the names of the function
        # print(f"lable: {hidden_layer_lable}") 
        # print(f"Learning rate: {gamma}") 
        # print(f"Penalty: {lam}")
        # print(f"Batch size: {M}")
        # print(f"Penalty method: {penalty_method}")
        # print(f"optimizer method: {optimizer_method}")
        # print("-"*80)

        CV_mse, CV_se,  = [], []
        train_mse = []
        CV_folds = []
        mean_fit_time = []

        for epoch in epochs: 
            # make pipeline with scaling and our own NN using the wrapper NeuralNetRegressor and scikit-learn
            model = make_pipeline(StandardScaler(), NeuralNetRegressor(layer_output_sizes=layer_output_size, activation_funcs=activation_func,
                    activation_ders=activation_der, cost_fun=mse, cost_der=mse_der, epochs=epoch, M=M, gamma=gamma, optimizer_method=optimizer_method, penalty_method=penalty_method, lam=lam))
            
            # cross-validate the pipeline using scikit learn
            scores = cross_validate(model, x_train, y_train, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)

            # get cv and training scores 
            CV_scores = -scores["test_score"] 
            train_scores = -scores["train_score"]
            fit_time = scores["fit_time"] # fit time for each fold
            # store all fold values
            CV_folds.append(CV_scores)

            # calculate and store cv mse, cv se and training mse
            CV_mse.append(np.mean(CV_scores)) 
            CV_se.append(np.std(CV_scores, ddof=1) / np.sqrt(len(CV_scores))) 
            train_mse.append(np.mean(train_scores)) 

            mean_fit_time.append(np.mean(fit_time)) # store mean fit time  

        idx = np.argmin(CV_mse) # find index of min value
        penalty_label = penalty_method if lam > 0 else "No penalty" # set penalty lable in print

        # print min value of all epochs with corresponding number of hidden layer, penalty and mean fit time
        print(f"MIN | hidden layers={hidden_layer_lable} | penalty={penalty_label} | lambda={lam} | best epochs={epochs[idx]} | CV MSE={CV_mse[idx]:.6f} pm {CV_se[idx]:.6f}| Mean fit time: {mean_fit_time[idx]:.6f} s")
        
        # print epoch, mse and se values and mean time to fit. Used to pick out with 1SE rule.
        for epoch, mse_value, se_value, fit_time in zip(epochs, CV_mse, CV_se, mean_fit_time):
            print(f"| epochs={epoch} | CV MSE={mse_value:.6f} pm {se_value:.6f} | Mean fit time: {fit_time:.6f} s")

        ax.errorbar(epochs, CV_mse, yerr=CV_se, fmt="o-", color="#004488", capsize=5, linewidth=3) # plot CV mse ,se
        ax.plot(epochs, train_mse, "s--", color="#BB5566", linewidth=3) # plot training mse

        CV_folds = np.asarray(CV_folds) # convert fold list to array (6, 5) (len of epoch interval, number of folds)
        # plot all 5 fold for a given epoch interval
        for fold in range(CV_folds.shape[1]): 
            ax.plot(epochs, CV_folds[:, fold], "-", color="#004488", alpha=0.3, linewidth=1) # plot folds

        ax.axhline(MSE, color=color_, ls=":", linewidth=3, label=rf"CV ({lable_})") # plot CV (OLS, Ridge or )
        ax.axhspan(MSE - SE, MSE + SE, color=color_, alpha=0.08) # plot SE (OLS)
        ax.axhline(0.01, color="black", ls=":",linewidth=3) # plot noise floor sigma = 0.01
        ax.axhline(var_y, color="#777777", ls=":",linewidth=3) # plot variance in observations var(y) = 0.94
        ax.set_title(hidden_layer_lable, fontsize=22)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=22)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Epochs", fontsize=24)
    # remove legends
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    # add legend to lower left plot
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", borderaxespad=0,fontsize=20, framealpha=0.95)
    fig.supylabel(r"MSE $\pm$ SE", fontsize=24)
    plt.show()

#---------------------------------------------------------------------
""" plot CV MSE and SE as a function of epochs for untuned NN with without regularization and CV MSE and SE for (OLS).
    No penalty, gamma=0.1, full batch, using "PLAIN" gradient descent."""

# optimal parameter values to CV as a function on epochs
parameters_50 = [0.0, 0.1, None]          # [penalty, learning rate, batchsize] for hidden layer [50]
parameters_100_100 = [0.0, 0.1, None]  # [penalty, learning rate, batchsize] for hidden layers [100, 100] 

# plot_1x2_MSE_epochs_(MSE=0.027, SE=0.007, parameters_50=parameters_50, parameters_100_100=parameters_100_100, optimizer_method="PLAIN") 

#---------------------------------------------------------------------

"""
LLM-assisted
------------
Tool: OpenAI Codex (GPT-6 Astra) (Novermber 2026)
Role: generated plotting code for 1x3 plot with additonal code for correcting placement of legend and lables.
Modifications: adapted the plotting code to our existing loops.
"""

def learning_rate_batch_plot(layer_output_size, activation_func, activation_der):
    """ CV MSE as a function of learning rate for full batch and SGD with batch sizes 10, 5 and 1, 
    using three different optimization methods (Plain gradient decent, RMSprop and Adam) using 2000 epochs.
    prnit min MSE and SE values for a given batch size with corresponding learning rate."""

    # lables and colors
    methods = ["PLAIN", "RMSPROP", "ADAM"]
    lables_methods = ["Plain", "RMSprop", "Adam"]
    colors = ["#004488", "#70BB55", "#AA4499", "#DDAA33"]

    # Learning rates
    gammas = [0.001, 0.003, 0.01, 0.03, 0.1] 

    # Mini batch size 
    batch_sizes = [None, 10, 5, 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True, layout="constrained")

    for ax, method, lable_method in zip(axes.flat, methods, lables_methods):
        for M, color in zip(batch_sizes, colors):

            CV_mse, CV_se = [], []

            for gamma in gammas:
                # Make pipeline with standardization and our own NN with the wrapper NeuralNetRegressor
                model = make_pipeline(StandardScaler(), NeuralNetRegressor(layer_output_sizes=layer_output_size, activation_funcs=activation_func,
                        activation_ders=activation_der, cost_fun=mse, cost_der=mse_der, epochs=2000, M=M, optimizer_method=method ,gamma=gamma))
                
                # cross-validate the pipeline using scikit learn 
                scores = cross_validate(model, x_train, y_train, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)

                # get cv scores
                CV_scores = -scores["test_score"]

               # calculate and store cv mse and cv se 
                CV_mse.append(np.mean(CV_scores))
                CV_se.append(np.std(CV_scores, ddof=1) / np.sqrt(len(CV_scores)))

            label = "Full batch" if M is None else f"Batch size = {M}" # set lable to batch size of not full batch
            ax.errorbar(gammas, CV_mse, yerr=CV_se, fmt="o-" , color= color, capsize=5, linewidth=3, label=label) # plot cv mse, se

            # find index for min value for a give minibatch size and corresponding lable
            idx = np.argmin(CV_mse)
            batch_label = "Full batch" if M is None else M
            # print min value of result for every mini batch size for a given method and corresponding learning rate
            print(f"{lable_method}, batch={batch_label}:min MSE={CV_mse[idx]:.6f}, SE={CV_se[idx]:.6f}, gamma={gammas[idx]}")

        ax.axhline(0.01, color="black", ls=":",linewidth=3) # plot noise floor sigma = 0.01
        ax.axhline(0.027, color="red", ls=":", linewidth=3) # plot CV (OLS)
        ax.axhspan(0.027 - 0.007, 0.027 + 0.007, color="red", alpha=0.08) # plot SE (OLS)
        ax.axhline(var_y, color="#777777", ls=":",linewidth=3) # plot variance in observations var(y) = 0.94
        ax.set_title(lable_method, fontsize=20)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=22)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.2)

    # remove legends
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
     # add legend to lower left plot
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower left", fontsize=16, framealpha=0.95)
    fig.supxlabel(r"Learning rate $\gamma$", fontsize=24)
    fig.supylabel(r"CV MSE $\pm$ SE", fontsize=24)
    plt.show()

print("-"*80)


""" Plot for one hidden layer [50] and print min values with corresponding learning rate."""

# learning_rate_batch_plot([50, 1], [sigmoid, identity], [sigmoid_der, identity_der]) 


""" Plot for two hidden layer [100, 100] and print min values with corresponding learning rate."""

# learning_rate_batch_plot([100, 100, 1], [sigmoid, sigmoid, identity], [sigmoid_der, sigmoid_der, identity_der]) 


#---------------------------------------------------------------------


""" plot 1x2  Final model for tuned NN without regularization"""

# optimal parameter values to CV as a function on epochs
parameters_50 = [0.0, 0.1, 10]          # [penalty, learning rate, batchsize] for hidden layer [50]
parameters_100_100 = [0.0, 0.03, None]  # [penalty, learning rate, batchsize] for hidden layers [100, 100] 

plot_1x2_MSE_epochs_(MSE=0.027, SE=0.007, parameters_50=parameters_50, parameters_100_100=parameters_100_100) 


#------------------------
## Part c)
#------------------------


"""
LLM-assisted
------------
Tool: OpenAI Codex (GPT-6 Astra) (November 2026)
Role: generated plotting code for 1x3 plot with additonal code for correcting placement of legend and lables.
Modifications: adapted the plotting code to our existing loops.
"""

def number_of_hidden_layers_and_nodes_plot():
    """ Plot CV MSE pm CV SE as a function of number of hidden layers with 100 nodes per layer. 
    Optimization using ADAM with learning rate gamma = 0.03 and batch size M=None for 300 epochs. """

    n_layers = np.arange(1, 8) # number of hidden layers

    # types of activation functions
    activations = [sigmoid, ReLU, leaky_ReLU]
    activation_ders = [sigmoid_der, ReLU_der, leaky_ReLU_der]
    activation_lables = ["Sigmoid", "ReLU", "Leaky ReLU"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True, layout="constrained")

    for ax, activation, activation_der, lable in zip(axes.flat, activations, activation_ders, activation_lables):
        CV_mse, CV_se = [], []
        train_mse = []
        CV_folds = []

        for n in n_layers:
            # generate n number of hidden layers with corresponding activation function and derivative of activation functions
            layer_output_size = [100] * n + [1]
            activation_func_ = [activation] * n + [identity] # identity as the last activation function
            activation_ders_ = [activation_der] * n + [identity_der]

            """Check if implementation of loop output have correct values""" 
            # print(f"layer {layer_output_size}") 
            # print(f"Activation functions: {[func.__name__ for func in activation_func_]}") # loop over activation functions and get the names of the function
            # print(f"Activation derivative: {[der.__name__ for  der in activation_ders_]}") # loop over activation derivative and get the names of the function
    
            # Make pipeline with standardization and our own NN with the wrapper NeuralNetRegressor
            model = make_pipeline(StandardScaler(), NeuralNetRegressor(layer_output_sizes=layer_output_size, activation_funcs=activation_func_,
                    activation_ders=activation_ders_, cost_fun=mse, cost_der=mse_der, optimizer_method="ADAM", M=None, gamma=0.03, epochs=300))
            
            # cross-validate the pipeline using scikit learn 
            scores = cross_validate(model, x_train, y_train, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)

            # get cv and training scores
            CV_scores = -scores["test_score"]
            train_scores = -scores["train_score"]

             # calculate and store cv mse, cv se and traning mse 
            CV_mse.append(np.mean(CV_scores))
            CV_se.append(np.std(CV_scores, ddof=1) / np.sqrt(len(CV_scores)))
            train_mse.append(np.mean(train_scores))

            CV_folds.append(CV_scores) # store CV folds


        ax.errorbar(n_layers, CV_mse, yerr=CV_se, fmt="o-", color="#004488", capsize=5, linewidth=3, label="CV") # plot cv mse
        ax.plot(n_layers, train_mse, "s--", color="#BB5566", linewidth=3, label="Training") # plot train mse

        idx = np.argmin(CV_mse) # find index of min value
        CV_folds = np.asarray(CV_folds) # convert fold list to array
                # plot all 5 fold 
        for fold in range(CV_folds.shape[1]): # convert fold list to array (6, 5) (len of epoch interval, number of folds)
            ax.plot(n_layers, CV_folds[:, fold], "-", color="#004488", alpha=0.3, linewidth=1) # plot folds

        ax.axhline(0.027, color="red", ls=":", linewidth=3, label=f"CV (OLS)")
        ax.axhspan(0.027 - 0.007, 0.027 + 0.007, color="red", alpha=0.08)
        ax.axhline(0.01, color="black", ls=":",linewidth=3)
        ax.axhline(var_y, color="#777777", ls=":",linewidth=3)

        ax.set_title(lable, fontsize=20)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=20)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.2)
    # remove legends
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    # add legend to upper left corner of first plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", fontsize=16, framealpha=0.95)
    fig.supxlabel("Number of hidden layers", fontsize=24)
    fig.supylabel(r"MSE $\pm$ SE", fontsize=24)
    plt.show()

    #---------------------------------------------------------------------

    """ 
    Plot CV MSE pm CV SE as a function of number of nodes in a single hidden layer
    Optimization using ADAM with learning rate gamma = 0.03 and batch size M=None for 300 epochs. """

    n_nodes = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000] # number of nodes

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=True, layout="constrained")

    for ax, activation, activation_der, lable in zip(axes.flat, activations, activation_ders, activation_lables):
        CV_mse, CV_se = [], []
        train_mse = []
        CV_folds = []

        for n_node in n_nodes:
            # generate one hidden layer with flexible number of nodes with corresponding activation function and derivative
            layer_output_size = [n_node] + [1]
            activation_func_ = [activation] + [identity]
            activation_ders_ = [activation_der] + [identity_der]

            """Chack if implementation of layers and activations""" 
            # print(f"layer {layer_output_size}") 
            # print(f"Activation functions: {[func.__name__ for func in activation_func_]}") # loop over activation functions and get the names of the function
            # print(f"Activation derivative: {[der.__name__ for  der in activation_ders_]}") # loop over activation derivative and get the names of the function
    
            # make pipeline with standardization and our own neural network 
            model = make_pipeline(StandardScaler(), NeuralNetRegressor(layer_output_sizes=layer_output_size, activation_funcs=activation_func_,
                    activation_ders=activation_ders_, cost_fun=mse, cost_der=mse_der, optimizer_method="ADAM", M=None, gamma=0.03, epochs=300))

            # cross-validate using scikit learn
            scores = cross_validate(model, x_train, y_train, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)

            # get cv and training scores
            CV_scores = -scores["test_score"]
            train_scores = -scores["train_score"]

            # calculate and store cv mse, cv se and training mse
            CV_mse.append(np.mean(CV_scores))
            CV_se.append(np.std(CV_scores, ddof=1) / np.sqrt(len(CV_scores)))
            train_mse.append(np.mean(train_scores))
    
            CV_folds.append(CV_scores) # store CV folds

        ax.errorbar(n_nodes, CV_mse, yerr=CV_se, fmt="o-", color="#004488", capsize=5, linewidth=3, label="CV")
        ax.plot(n_nodes, train_mse, "s--", color="#BB5566", linewidth=2, label="Training")
        
        idx = np.argmin(CV_mse) # find index of min value
        CV_folds = np.asarray(CV_folds) # convert fold list to array
        print(f"Folds at minimum: {CV_folds[idx, :]}") # print corresponding min value for folds
        print("-"*80)

                # plot all 5 fold 
        for fold in range(CV_folds.shape[1]): # convert fold list to array (6, 5) (len of epoch interval, number of folds)
            ax.plot(n_nodes, CV_folds[:, fold], "-", color="#004488", alpha=0.3, linewidth=1) # plot folds


        ax.axhline(0.027, color="red", ls=":", linewidth=3, label=f"CV (OLS)")
        ax.axhspan(0.027 - 0.007, 0.027 + 0.007, color="red", alpha=0.08)
        ax.axhline(0.01, color="black", ls=":",linewidth=3) 
        ax.axhline(var_y, color="#777777", ls=":",linewidth=3)

        ax.set_title(lable, fontsize=20)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=20)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.2)

    # remove legends
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    # add legend to upper right corner of first plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", fontsize=16, framealpha=0.95)
    fig.supxlabel("Number of nodes (one hidden layer)", fontsize=24)
    fig.supylabel(r"MSE $\pm$ SE", fontsize=24)
    plt.show()

""" Plot mse as function of hidden layers and mse as function of number of nodes for one hidden layer"""
number_of_hidden_layers_and_nodes_plot() 

# ---------------------------------------------------------------------

# ------------------------
# Part d)
# ------------------------

# #---------------------------------------------------------------------
# """ Ridge CV mse and se """
# #---------------------------------------------------------------------
# Polynomial degrees
degrees = range(1, 16)
# interval of hyperparameter values used to find the optimal model for Ridge
lmbd_Ridge = np.linspace(1e-5, 1e-1, 30)

mse_Ridge, se_Ridge, time_fit_Ridge = polynomial_CV(x_train, y_train, method="Ridge", degrees=degrees, lmbd=lmbd_Ridge)
candidate_3, row, column = within_one_standard_error(mse_Ridge, se_Ridge, D2=True)
 

print("-"*80)
print("Ridge: Combinations of polynomial degree and regularisation that are within one standard error")
for i, j in zip(row, column):
    print(f"degree = {degrees[i]:>4}, lambda = {lmbd_Ridge[j]:>4.5f}, mse pm se:{mse_Ridge[i, j]:>4.4f} pm {se_Ridge[i, j]:>4.4f}, mean time to fit: {time_fit_Ridge[i, j]:>4.6f}")
print("-"*80)

# #---------------------------------------------------------------------
# """ Lasso CV mse and se """
# #---------------------------------------------------------------------

# interval of hyperparameter values used to find the optimal model for Lasso
lmbd_Lasso = np.linspace(1e-3, 1e-1, 30)


mse_Lasso, se_Lasso, time_fit_Lasso = polynomial_CV(x_train, y_train, method="Lasso", degrees=degrees, lmbd=lmbd_Lasso)
candidate_Lasso, row_Lasso, column_Lasso = within_one_standard_error(mse_Lasso, se_Lasso, D2=True)


print("-"*80)
print(" Lasso: Combinations of polynomial degree and regularisation that are within one standard error")
for i, j in zip(row_Lasso, column_Lasso):
    print(f"degree = {degrees[j]:>4}, lambda = {lmbd_Lasso[i]:>4.4f}, mse pm se:{mse_Lasso[i, j]:>4.4f} pm {se_Lasso[i, j]:>4.4f}, mean time to fit: {time_fit_Lasso[i, j]:.6f}")
print("-"*80)

# # #---------------------------------------------------------------------

def NN_CV(x_train, y_train, penalty_method="L2", gamma=[], lmbd=[], epochs=300, M=10, layer_output_sizes=[50, 1], activation_funcs=[sigmoid, identity], activation_ders=[sigmoid_der, identity_der]):

    """Performce grid search cross-validation for learning rates and penalty for L2 or L1."""

    if penalty_method=="L2": # Select L2 regularization
        param_grid = {"neuralnetregressor__gamma": gamma, "neuralnetregressor__lam": lmbd} # dictionary key and lists of parameters  

    elif penalty_method=="L1": # Select L1 regularization
        param_grid = {"neuralnetregressor__gamma": gamma, "neuralnetregressor__lam": lmbd} # dictionary key and lists of parameters  
        
    else:
        raise ValueError("Valid penalty_method names: L2 and L1")
    # make pipeline with standardization and our own NN
    net = make_pipeline(StandardScaler(), NeuralNetRegressor(layer_output_sizes=layer_output_sizes, activation_funcs=activation_funcs,
                activation_ders=activation_ders, cost_fun=mse, cost_der=mse_der, epochs=epochs, optimizer_method="ADAM", M=M, penalty_method=penalty_method))
    
    # performe grid search CV on pipeline using scikit learn
    search = GridSearchCV(net, param_grid, cv=cv, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
    search.fit(x_train, y_train) # fit training data

    i, j = len(gamma), len(lmbd) #(gamma, lambda)

    # get matrix with cv mse and se, and correspinding mean time to fit
    results = search.cv_results_ # CV MSE
    CV_mse = -results["mean_test_score"].reshape(i, j) # CV MSE
    CV_se = results["std_test_score"].reshape(i, j) / np.sqrt(cv.get_n_splits() - 1) # CV SE: std_test_score uses ddof=0: s/sqrt(k) = std/sqrt(k-1)
    time = results["mean_fit_time"].reshape(i, j) # mean fit time [s]

    # return CV mse, se and neab time to fit
    return CV_mse, CV_se, time  

#---------------------------------------------------------------------

# parameters used for grid search
gamma=[0.001, 0.003, 0.01, 0.03, 0.1] # Learning rate
lmbd=[0.0, 1e-5, 1e-4, 1e-3, 1e-2] # Hyperparameter

"""estimate cv mse and se from grid search for learning rate and L2 regularizaton for one hidden layer [50]."""

mse_L2, se_L2, time_L2 = NN_CV(x_train, y_train, penalty_method="L2", gamma=gamma, lmbd=lmbd, epochs=300, M=10)
# find candidates within one standard error
cand_L2, i_L2, j_L2 = within_one_standard_error(mse_L2, se_L2, D2=True)

print("-"*80)
print("(L2): Combinations of learning rate and L2 regularization that are within one standard")
for i, j in zip(i_L2, j_L2):
    print(f"gamma = {gamma[i]}, lambda = {lmbd[j]}, mse pm se:{mse_L2[i, j]:.5f} pm {se_L2[i, j]:5f}, mean time to fit = {time_L2[i, j]:.6f}")
print("-"*80)

#---------------------------------------------------------------------
"""estimate cv mse and se from grid search for learning rate and L1 regularizaton for one hidden layer [50]."""

mse_L1, se_L1, time_L1 = NN_CV(x_train, y_train, penalty_method="L1", gamma=gamma, lmbd=lmbd, epochs=300, M=10)
# find candidates within one standard error
cand_L1, id_L1, jd_L1 = within_one_standard_error(mse_L1, se_L1, D2=True)

print("-"*80)
print("(L1): Combinations of learning rate and L1 regularization that are within one standard")
for i, j in zip(id_L1, jd_L1):
    print(f"gamma = {gamma[i]}, lambda = {lmbd[j]}, mse pm se:{mse_L1[i, j]:.5f} pm {se_L1[i, j]:5f}, mean time to fit = {time_L1[i, j]:.6f}")
print("-"*80)

#---------------------------------------------------------------------
"""estimate cv mse and se from grid search for learning rate and L2 regularizaton for two hidden layers [100, 100]."""

mse_L2_100, se_L2_100, time_L2_100 = NN_CV(x_train, y_train, penalty_method="L2", gamma=gamma, lmbd=lmbd, epochs=300, M=None, 
                                layer_output_sizes=[100, 100, 1], activation_funcs=[sigmoid, sigmoid, identity], activation_ders=[sigmoid_der, sigmoid_der, identity_der])
# find candidates within one standard error
cand_L2_100, i_L2_100, j_L2_100 = within_one_standard_error(mse_L2_100, se_L2_100, D2=True)


print("-"*80)
print("(L2): Combinations of learning rate and L2 regularization that are within one standard")
for i, j in zip(i_L2_100, j_L2_100):
    print(f"gamma = {gamma[i]}, lambda = {lmbd[j]}, mse pm se:{mse_L2_100[i, j]:.5f} pm {se_L2_100[i, j]:5f}, mean time to fit = {time_L2_100[i, j]:.6f}")
print("-"*80)

#---------------------------------------------------------------------
"""estimate cv mse and se from grid search for learning rate and L1 regularizaton for two hidden layers [100, 100]."""

mse_L1_100, se_L1_100, time_L1_100 = NN_CV(x_train, y_train, penalty_method="L1", gamma=gamma, lmbd=lmbd, epochs=300, M=None, 
                               layer_output_sizes=[100, 100, 1], activation_funcs=[sigmoid, sigmoid, identity], activation_ders=[sigmoid_der, sigmoid_der, identity_der])
# find candidates within one standard error
cand_L1_100, id_L1_100, jd_L1_100 = within_one_standard_error(mse_L1_100, se_L1_100, D2=True)


print("-"*80)
print("(L1): Combinations of learning rate and L1 regularization that are within one standard")
for i, j in zip(id_L1_100, jd_L1_100):
    print(f"gamma = {gamma[i]}, lambda = {lmbd[j]}, mse pm se:{mse_L1_100[i, j]:.5f} pm {se_L1_100[i, j]:5f}, mean time to fit = {time_L1_100[i, j]:.6f}")
print("-"*80)

# ---------------------------------------------------------------------

def heat_plot(mse_L, cand_L, penalty_method="L2", hidden_layers="[50]"):
    """
    Find cross-validation MSE and SE within one standard error of the minimum MSE and corresponding indices.
    """

    # Modified from https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week44/week44tuesday.ipynb
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    im = ax.imshow(mse_L, cmap="YlGnBu", origin="lower")

    ax.set_xticks(range(len(lmbd)))
    ax.set_xticklabels([f"{v:g}" for v in lmbd], fontsize=20)
    ax.set_yticks(range(len(gamma)))
    ax.set_yticklabels([f"{v:g}" for v in gamma], fontsize=20)
    ax.set_xlabel(r"$\lambda$", fontsize=24)
    ax.set_ylabel(r"$\gamma$", fontsize=24)

    for i in range(len(gamma)):
        for j in range(len(lmbd)):
            ax.text(j, i, f"{mse_L[i, j]:.3f}", ha="center", va="center", fontsize=22, color="white" if mse_L[i, j] > 0.6 * mse_L.max() else "black",
                fontweight="bold" if cand_L[i, j] else "normal")
            
    ax.set_title(f"({penalty_method}), {hidden_layers}, CV MSE; bold = within 1 SE", fontsize=24)
    ax.grid(False)
    plt.show()

#---------------------------------------------------------------------

""" heat plot with L2 regularization and one hidden layer [50]"""
heat_plot(mse_L2, cand_L2, penalty_method="L2", hidden_layers="[50]") 

""" heat plot with L1 regularization and one hidden layer [50]"""
heat_plot(mse_L1, cand_L1, penalty_method="L1", hidden_layers="[50]")

""" heat plot with L2 regularization and two hidden layers [100, 100]"""
heat_plot(mse_L2_100, cand_L2_100, penalty_method="L2", hidden_layers="[100, 100]") 

""" heat plot with L1 regularization and two hidden layers [100, 100]"""
heat_plot(mse_L1_100, cand_L1_100, penalty_method="L1", hidden_layers="[100, 100]") 

#---------------------------------------------------------------------

""" (1x2) plot CV MSE and SE as a function of epochs for gridsearched learning rate and L2 penalty, plotted for two
configurations [50] and [100, 100] and CV MSE of Ridge."""

# optimal parameter values to CV as a function on epochs
parameters_50_L2 = [0.0001, 0.1, 10]          # [penalty, learning rate, batchsize] for hidden layer [50]
parameters_100_100_L2 = [0.0001, 0.03, None]  # [penalty, learning rate, batchsize] for hidden layers [100, 100]

plot_1x2_MSE_epochs_(penalty_method="L2", MSE=0.027, SE=0.006, parameters_50=parameters_50_L2, parameters_100_100=parameters_100_100_L2) 

#---------------------------------------------------------------------

""" plot CV MSE and SE as a function of epochs for gridsearched learning rate and L1 penalty, plotted for two
configurations [50] and [100, 100] and CV MSE of Lasso."""

# optimal parameter values to CV as a function on epochs
parameters_50_L1 = [0.001, 0.1, 10]          # [penalty, learning rate, batchsize] for hidden layer [50]
parameters_100_100_L1 = [0.0001, 0.03, None]  # [penalty, learning rate, batchsize] for hidden layers [100, 100]

plot_1x2_MSE_epochs_(penalty_method="L1", MSE=0.029, SE=0.004, parameters_50=parameters_50_L1, parameters_100_100=parameters_100_100_L1) 

#---------------------------------------------------------------------


#------------------------
## Part d)
## Final models fitted on test data
#------------------------


def test_MSE(x_train, y_train, x_test, y_test, lam=0.0, penalty="L2", gamma=0.01, epochs=1000, M=10, layer_output_sizes=[50, 1], activation_funcs=[sigmoid, identity], activation_ders=[sigmoid_der, identity_der]):
    """
    Calculate test MSE from trained model 
    """
    network_input_size = 1
    
    # use scikit-learn to standardize the data
    scaler = StandardScaler()
    Xs = scaler.fit_transform(x_train) # fit traning data
    X_test = scaler.transform(x_test) # scale test data to training data
    # make instance of our own NN class
    net = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun=mse, cost_der=mse_der)
    net.optimizer("ADAM") # select Adam as optimizer
    net.learning_rate(gamma) # set learning rate

    if lam != 0.0: # add penalty to NN
        net.add_penalty(lam, penalty)
    # train network 
    net.train(Xs, y_train, epochs=epochs, M=M)
    # get prediction 
    y_pred_test = np.asarray(net.predict(X_test)).ravel()
    # test values
    y_true_test = np.asarray(y_test).ravel()
    # estimate test MSE
    MSE_test = np.mean((y_pred_test - y_true_test)**2)
    # estimate corresponding standard error
    SE_test = np.std((y_pred_test - y_true_test)**2, ddof=1) / np.sqrt(len(y_pred_test))
    # count number of parameters for a given configuration
    total_parameters = net.parameter_count() 

    print(f"({penalty}) Test-MSE pm SE: {MSE_test:.5f} pm {SE_test:.5f}, {layer_output_sizes}, parameters {total_parameters}")
    print("-"*80)

# ---------------------------------------------------------------------

""" test MSE for final model for one hidden layer [50]. """

# Without penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.0, penalty="No penalty", gamma=0.1, epochs=300, M=10, 
         layer_output_sizes=[50, 1], activation_funcs=[sigmoid, identity], activation_ders=[sigmoid_der, identity_der])

# L2 penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.0001, penalty="L2", gamma=0.1, epochs=300, M=10, 
         layer_output_sizes=[50, 1], activation_funcs=[sigmoid, identity], activation_ders=[sigmoid_der, identity_der])

# L1 penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.001, penalty="L1", gamma=0.1, epochs=300, M=10, 
         layer_output_sizes=[50, 1], activation_funcs=[sigmoid, identity], activation_ders=[sigmoid_der, identity_der])

# ---------------------------------------------------------------------

""" test MSE for final model for two hidden layers [100, 100]. """

# Without penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.0, penalty="No penalty", gamma=0.03, epochs=300, M=None, 
         layer_output_sizes=[100, 100, 1], activation_funcs=[sigmoid, sigmoid, identity], activation_ders=[sigmoid_der, sigmoid_der, identity_der])

# L2 penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.0001, penalty="L2", gamma=0.03, epochs=300, M=None, 
         layer_output_sizes=[100, 100, 1], activation_funcs=[sigmoid, sigmoid, identity], activation_ders=[sigmoid_der, sigmoid_der, identity_der])

# L1 penalty
test_MSE(x_train, y_train, x_test, y_test, lam=0.0001, penalty="L1", gamma=0.03, epochs=300, M=None, 
         layer_output_sizes=[100, 100, 1], activation_funcs=[sigmoid, sigmoid, identity], activation_ders=[sigmoid_der, sigmoid_der, identity_der])

# ---------------------------------------------------------------------

mse_OLS, se_OLS, time_fit_OLS = polynomial_CV(x_train, y_train, method="OLS", degrees=range(1, 16))
candidate_OLS, index_OLS = within_one_standard_error(mse_OLS, se_OLS)

print("OLS: Polynomial degrees that are within one standard error")
for i in index_OLS[0]:
    print(f"Polynomial degree: {i + 1}, CV MSE: {mse_OLS[i]:>5.3f} pm {se_OLS[i]:>5.3f}, mean time to fit: {time_fit_OLS[i]:>5.6f}")
print("-"*80)

# ---------------------------------------------------------------------

def fit_polynomial_test_mse(x_train, y_train, x_test, y_test, lam, degree, method):
    """ Estimate finale test MSE and SE for OLS, Rindge and Lasso"""
    # make pipeline with polynomial features standardization and regression method:
    if method=="OLS": # Select OLS
        poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LinearRegression())

    elif method=="Ridge": # Select Ridge
        poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), Ridge(len(x_train) * lam))

    elif method=="Lasso": # Select Lasso
        poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), Lasso(lam / 2, max_iter=100000, tol=1e-5))

    else:
        raise ValueError("Valid method names: OLS, Ridge, Lasso")

    # fit data to molynomial pipeline
    model = poly.fit(x_train, y_train)
    # make prediction with test data
    y_pred = poly.predict(x_test).ravel()
    y_true = np.asarray(y_test).ravel()
    # calculate and return test MSE and corresponding SE
    MSE_test = np.mean((y_pred - y_true)**2)
    SE_test = np.std((y_pred - y_true)**2, ddof=1) / np.sqrt(len(y_true))

    print(f"test MSE ({method}): {MSE_test:.3f} pm {SE_test:.3f}")
    print("-"*80)



# """ print final test MSE and SE regression methods"""
# # OLS
fit_polynomial_test_mse(x_train, y_train, x_test, y_test, lam=0.0, degree=8, method="OLS")
# # Ridge
fit_polynomial_test_mse(x_train, y_train, x_test, y_test, lam=0.00001, degree=8, method="Ridge")
# # Lasso
fit_polynomial_test_mse(x_train, y_train, x_test, y_test, lam=0.001, degree=6, method="Lasso")

# ---------------------------------------------------------------------






""" 
                                ##############################
                                ## Print out and key values ##
                                ##############################

### (1) 

Untuned nettwork min MSE values: 
--------------------------------------------------------------------------------
| hidden layers=[50] | penalty=No penalty | lambda=0 | best epochs=3000 | CV MSE=0.022973 pm 0.003754|
Folds at minimum: [0.01973123 0.01700893 0.01430174 0.03131757 0.03250773]
--------------------------------------------------------------------------------
| hidden layers=[100, 100] | penalty=No penalty | lambda=0 | best epochs=3000 | CV MSE=0.093219 pm 0.014613|
Folds at minimum: [0.12283096 0.04922868 0.0720624  0.12477161 0.09720213]
--------------------------------------------------------------------------------




## (2) 

CV MSE as a function of learning rate for full batch and SGD with batch sizes 10, 5 and 1, 
using three different optimization methods (Plain gradient decent, RMSprop and Adam) trained for 2000 epochs.

# ---------------------------------------------------------------------

### Min values for each optimization method for hidden layer [50]:

Plain, batch=Full batch:min MSE=0.025155, SE=0.003755, gamma=0.1
Plain, batch=10:min MSE=0.021366, SE=0.004597, gamma=0.1
Plain, batch=5:min MSE=0.019648, SE=0.004621, gamma=0.1
Plain, batch=1:min MSE=0.019835, SE=0.004760, gamma=0.03
RMSprop, batch=Full batch:min MSE=0.019773, SE=0.003708, gamma=0.001
RMSprop, batch=10:min MSE=0.018054, SE=0.004168, gamma=0.001
RMSprop, batch=5:min MSE=0.017785, SE=0.004099, gamma=0.001
RMSprop, batch=1:min MSE=0.017984, SE=0.004235, gamma=0.001
Adam, batch=Full batch:min MSE=0.017598, SE=0.003637, gamma=0.01
(Adam, batch=10:min MSE=0.017177, SE=0.003520, gamma=0.01) <-------- min MSE
Adam, batch=5:min MSE=0.017597, SE=0.003785, gamma=0.001
Adam, batch=1:min MSE=0.018253, SE=0.003987, gamma=0.001

# ---------------------------------------------------------------------

### Min values for each optimization method for hidden layers [100, 100]:

Plain, batch=Full batch:min MSE=0.051337, SE=0.012379, gamma=0.03
Plain, batch=10:min MSE=0.018808, SE=0.004073, gamma=0.03
Plain, batch=5:min MSE=0.018012, SE=0.003990, gamma=0.03
Plain, batch=1:min MSE=0.017658, SE=0.003484, gamma=0.1
RMSprop, batch=Full batch:min MSE=0.018020, SE=0.003860, gamma=0.003
RMSprop, batch=10:min MSE=0.017265, SE=0.003558, gamma=0.001
RMSprop, batch=5:min MSE=0.017468, SE=0.003570, gamma=0.001
RMSprop, batch=1:min MSE=0.017475, SE=0.003250, gamma=0.001
Adam, batch=Full batch:min MSE=0.017148, SE=0.003553, gamma=0.03 <-------- min MSE
Adam, batch=10:min MSE=0.017583, SE=0.003606, gamma=0.001
Adam, batch=5:min MSE=0.017416, SE=0.003558, gamma=0.001
Adam, batch=1:min MSE=0.017921, SE=0.003189, gamma=0.001

# ---------------------------------------------------------------------
!! 1SE values for learning rate picked out visualy from plot. 
# ---------------------------------------------------------------------





## (3)
Results from grid search for polynomial degree and hyperparameter lambda for Ridge and Lasso
--------------------------------------------------------------------------------
OLS: Polynomial degrees that are within one standard error
Polynomial degree: 8, CV MSE: 0.027 pm 0.007, mean time to fit: 0.000632 
--------------------------------------------------------------------------------
Ridge: Combinations of polynomial degree and regularisation that are within one standard error
degree =    8, lambda = 0.00001, mse pm se:0.0269 pm 0.0059, mean time to fit: 0.000631   <-------- 1SE
degree =    9, lambda = 0.00001, mse pm se:0.0247 pm 0.0052, mean time to fit: 0.000612   <-------- min MSE
degree =   10, lambda = 0.00001, mse pm se:0.0285 pm 0.0074, mean time to fit: 0.000642
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 Lasso: Combinations of polynomial degree and regularisation that are within one standard error
degree =    6, lambda = 0.0010, mse pm se:0.0290 pm 0.0042, mean time to fit: 0.001013    <---------1SE
degree =    7, lambda = 0.0010, mse pm se:0.0313 pm 0.0057, mean time to fit: 0.001081    <-------- min MSE
degree =    8, lambda = 0.0010, mse pm se:0.0316 pm 0.0056, mean time to fit: 0.003271
degree =    8, lambda = 0.0044, mse pm se:0.0331 pm 0.0053, mean time to fit: 0.000658    
degree =   10, lambda = 0.0044, mse pm se:0.0329 pm 0.0053, mean time to fit: 0.000661
degree =   11, lambda = 0.0044, mse pm se:0.0330 pm 0.0053, mean time to fit: 0.000664
--------------------------------------------------------------------------------

# ---------------------------------------------------------------------





## (4)
Best values for tuned NN without penalty used to choose epochs

# ---------------------------------------------------------------------

MIN | hidden layers=[50] | penalty=No penalty | lambda=0.0 | best epochs=300 | CV MSE=0.021173 pm 0.004931| Mean fit time: 0.082533 s
Folds at minimum: [0.01709397 0.0194977  0.00981715 0.01994061 0.03951363]
| epochs=50 | CV MSE=0.029374 pm 0.006168 | Mean fit time: 0.014043 s
| epochs=100 | CV MSE=0.025960 pm 0.006529 | Mean fit time: 0.027404 s
| epochs=300 | CV MSE=0.021173 pm 0.004931 | Mean fit time: 0.082533 s <---------1SE
| epochs=1000 | CV MSE=0.021472 pm 0.004760 | Mean fit time: 0.273348 s
| epochs=2000 | CV MSE=0.023383 pm 0.003410 | Mean fit time: 0.542998 s
| epochs=3000 | CV MSE=0.023902 pm 0.006221 | Mean fit time: 0.808417 s

--------------------------------------------------------------------------------
MIN | hidden layers=[100, 100] | penalty=No penalty | lambda=0.0 | best epochs=1000 | CV MSE=0.016923 pm 0.003670| Mean fit time: 0.340153 s
Folds at minimum: [0.0132405  0.01437807 0.00902962 0.01737404 0.03059096]

| epochs=50 | CV MSE=0.097048 pm 0.016193 | Mean fit time: 0.017614 s
| epochs=100 | CV MSE=0.083953 pm 0.015706 | Mean fit time: 0.033728 s
| epochs=300 | CV MSE=0.017316 pm 0.003970 | Mean fit time: 0.101549 s <---------1SE 
| epochs=1000 | CV MSE=0.016923 pm 0.003670 | Mean fit time: 0.340153 s
| epochs=2000 | CV MSE=0.017148 pm 0.003553 | Mean fit time: 0.677082 s
| epochs=3000 | CV MSE=0.017725 pm 0.003467 | Mean fit time: 1.017794 s

# ---------------------------------------------------------------------







## (5)
# Grid search of learning rate and penalty for L2 regularization [50]
# Picked out with grid search plot using 1SE rule
--------------------------------------------------------------------------------
(L2): Combinations of learning rate and L2 regularization that are within one standard
gamma = 0.003, lambda = 0.0, mse pm se:0.01885 pm 0.004005, mean time to fit = 0.125433 <--------- MIN 
gamma = 0.003, lambda = 1e-05, mse pm se:0.01921 pm 0.004008, mean time to fit = 0.126874
gamma = 0.003, lambda = 0.0001, mse pm se:0.02222 pm 0.004134, mean time to fit = 0.117829
gamma = 0.01, lambda = 0.0, mse pm se:0.01943 pm 0.005100, mean time to fit = 0.122878
gamma = 0.01, lambda = 1e-05, mse pm se:0.01947 pm 0.005028, mean time to fit = 0.117763
gamma = 0.01, lambda = 0.0001, mse pm se:0.02080 pm 0.004648, mean time to fit = 0.120252
gamma = 0.03, lambda = 0.0, mse pm se:0.01961 pm 0.004511, mean time to fit = 0.121829
gamma = 0.03, lambda = 1e-05, mse pm se:0.01988 pm 0.005349, mean time to fit = 0.125630
gamma = 0.03, lambda = 0.0001, mse pm se:0.02236 pm 0.006169, mean time to fit = 0.125187 
gamma = 0.1, lambda = 0.0, mse pm se:0.02117 pm 0.004931, mean time to fit = 0.116631
gamma = 0.1, lambda = 1e-05, mse pm se:0.02141 pm 0.004938, mean time to fit = 0.120414
gamma = 0.1, lambda = 0.0001, mse pm se:0.02258 pm 0.004947, mean time to fit = 0.119644 <---------1SE 
--------------------------------------------------------------------------------


# Grid search of learning rate and penalty for L2 regularization [100, 100]
--------------------------------------------------------------------------------
(L2): Combinations of learning rate and L2 regularization that are within one standard
gamma = 0.003, lambda = 0.0, mse pm se:0.01725 pm 0.003705, mean time to fit = 0.149077
gamma = 0.003, lambda = 1e-05, mse pm se:0.01752 pm 0.003674, mean time to fit = 0.148719
gamma = 0.01, lambda = 0.0, mse pm se:0.01697 pm 0.003951, mean time to fit = 0.150567 <--------- MIN 
gamma = 0.01, lambda = 1e-05, mse pm se:0.01732 pm 0.003814, mean time to fit = 0.150941
gamma = 0.03, lambda = 0.0, mse pm se:0.01732 pm 0.003970, mean time to fit = 0.145282
gamma = 0.03, lambda = 1e-05, mse pm se:0.01756 pm 0.003707, mean time to fit = 0.149094
gamma = 0.03, lambda = 0.0001, mse pm se:0.02002 pm 0.003601, mean time to fit = 0.142861 <---------1SE 
--------------------------------------------------------------------------------

# Choosen epochs for the two configurations with L2 penalty
--------------------------------------------------------------------------------
MIN | hidden layers=[50] | penalty=L2 | lambda=0.0001 | best epochs=2000 | CV MSE=0.020439 pm 0.003874| Mean fit time: 0.534613 s
| epochs=50 | CV MSE=0.028906 pm 0.004303 | Mean fit time: 0.014243 s
| epochs=100 | CV MSE=0.036021 pm 0.007500 | Mean fit time: 0.027836 s
| epochs=300 | CV MSE=0.022576 pm 0.004947 | Mean fit time: 0.082513 s <---------1SE 
| epochs=1000 | CV MSE=0.021859 pm 0.003244 | Mean fit time: 0.271306 s
| epochs=2000 | CV MSE=0.020439 pm 0.003874 | Mean fit time: 0.534613 s <--------- MIN 
| epochs=3000 | CV MSE=0.026821 pm 0.007858 | Mean fit time: 0.806657 s
Folds at minimum: [0.01639605 0.01123296 0.01554832 0.02720733 0.0318085 ]
--------------------------------------------------------------------------------
MIN | hidden layers=[100, 100] | penalty=L2 | lambda=0.0001 | best epochs=2000 | CV MSE=0.018493 pm 0.003693| Mean fit time: 0.669721 s
| epochs=50 | CV MSE=0.097787 pm 0.016248 | Mean fit time: 0.017776 s
| epochs=100 | CV MSE=0.095752 pm 0.016058 | Mean fit time: 0.033807 s
| epochs=300 | CV MSE=0.020024 pm 0.003601 | Mean fit time: 0.101048 s <---------1SE 
| epochs=1000 | CV MSE=0.019115 pm 0.003424 | Mean fit time: 0.336597 s
| epochs=2000 | CV MSE=0.018493 pm 0.003693 | Mean fit time: 0.669721 s <--------- MIN 
| epochs=3000 | CV MSE=0.018559 pm 0.003783 | Mean fit time: 0.997411 s
Folds at minimum: [0.01545291 0.01444005 0.00963972 0.02190772 0.03102634]
--------------------------------------------------------------------------------





## (6)

# Grid search of learning rate and penalty for L1 regularization [50]
# Picked out with grid search plot using 1SE rule
--------------------------------------------------------------------------------
(L1): Combinations of learning rate and L1 regularization that are within one standard
gamma = 0.003, lambda = 0.0, mse pm se:0.01885 pm 0.004005, mean time to fit = 0.126947 <--------- MIN 
gamma = 0.003, lambda = 1e-05, mse pm se:0.01891 pm 0.004011, mean time to fit = 0.124326
gamma = 0.003, lambda = 0.0001, mse pm se:0.01931 pm 0.004032, mean time to fit = 0.123850
gamma = 0.01, lambda = 0.0, mse pm se:0.01943 pm 0.005100, mean time to fit = 0.127432
gamma = 0.01, lambda = 1e-05, mse pm se:0.01931 pm 0.005037, mean time to fit = 0.124303
gamma = 0.01, lambda = 0.0001, mse pm se:0.01864 pm 0.004720, mean time to fit = 0.124063
gamma = 0.01, lambda = 0.001, mse pm se:0.02062 pm 0.004485, mean time to fit = 0.122385
gamma = 0.03, lambda = 0.0, mse pm se:0.01961 pm 0.004511, mean time to fit = 0.120292
gamma = 0.03, lambda = 1e-05, mse pm se:0.01934 pm 0.004566, mean time to fit = 0.118349
gamma = 0.03, lambda = 0.0001, mse pm se:0.01999 pm 0.005823, mean time to fit = 0.126577
gamma = 0.03, lambda = 0.001, mse pm se:0.02137 pm 0.004620, mean time to fit = 0.118973
gamma = 0.1, lambda = 0.0, mse pm se:0.02117 pm 0.004931, mean time to fit = 0.122211
gamma = 0.1, lambda = 1e-05, mse pm se:0.02141 pm 0.004984, mean time to fit = 0.125953
gamma = 0.1, lambda = 0.0001, mse pm se:0.02120 pm 0.004760, mean time to fit = 0.120806
gamma = 0.1, lambda = 0.001, mse pm se:0.02273 pm 0.004955, mean time to fit = 0.114857 <---------1SE 
--------------------------------------------------------------------------------

# Grid search of learning rate and penalty for L1 regularization [100, 100]

--------------------------------------------------------------------------------
(L1): Combinations of learning rate and L1 regularization that are within one standard
gamma = 0.003, lambda = 0.0, mse pm se:0.01725 pm 0.003705, mean time to fit = 0.152129
gamma = 0.003, lambda = 1e-05, mse pm se:0.01743 pm 0.003666, mean time to fit = 0.158514
gamma = 0.003, lambda = 0.0001, mse pm se:0.01992 pm 0.003567, mean time to fit = 0.158818
gamma = 0.01, lambda = 0.0, mse pm se:0.01697 pm 0.003951, mean time to fit = 0.153509 <--------- MIN 
gamma = 0.01, lambda = 1e-05, mse pm se:0.01705 pm 0.003896, mean time to fit = 0.163780
gamma = 0.01, lambda = 0.0001, mse pm se:0.01895 pm 0.003708, mean time to fit = 0.164276
gamma = 0.03, lambda = 0.0, mse pm se:0.01732 pm 0.003970, mean time to fit = 0.150038
gamma = 0.03, lambda = 1e-05, mse pm se:0.01745 pm 0.003926, mean time to fit = 0.150324
gamma = 0.03, lambda = 0.0001, mse pm se:0.01816 pm 0.003795, mean time to fit = 0.152532 <---------1SE 
gamma = 0.1, lambda = 1e-05, mse pm se:0.01810 pm 0.004039, mean time to fit = 0.139042 
--------------------------------------------------------------------------------



--------------------------------------------------------------------------------
layer [50, 1]
Activation functions: ['sigmoid', 'identity']
Activation derivative: ['sigmoid_der', 'identity_der']
lable: [50]
Learning rate: 0.1
Penalty: 0.001
Batch size: 10
Penalty method: L1
optimizer method: ADAM
--------------------------------------------------------------------------------
MIN | hidden layers=[50] | penalty=L1 | lambda=0.001 | best epochs=2000 | CV MSE=0.020769 pm 0.004071| Mean fit time: 0.553938 s
| epochs=50 | CV MSE=0.026334 pm 0.003597 | Mean fit time: 0.014298 s
| epochs=100 | CV MSE=0.035140 pm 0.005847 | Mean fit time: 0.027760 s
| epochs=300 | CV MSE=0.022727 pm 0.004955 | Mean fit time: 0.084298 s  <--------- 1SE
| epochs=1000 | CV MSE=0.022438 pm 0.003313 | Mean fit time: 0.275982 s
| epochs=2000 | CV MSE=0.020769 pm 0.004071 | Mean fit time: 0.553938 s <--------- MIN 
| epochs=3000 | CV MSE=0.026682 pm 0.007465 | Mean fit time: 0.820836 s
layer [100, 100, 1]
Activation functions: ['sigmoid', 'sigmoid', 'identity']
Activation derivative: ['sigmoid_der', 'sigmoid_der', 'identity_der']
lable: [100, 100]
Learning rate: 0.03
Penalty: 0.0001
Batch size: None
Penalty method: L1
optimizer method: ADAM
--------------------------------------------------------------------------------
MIN | hidden layers=[100, 100] | penalty=L1 | lambda=0.0001 | best epochs=2000 | CV MSE=0.017932 pm 0.004010| Mean fit time: 0.747514 s
| epochs=50 | CV MSE=0.097701 pm 0.016242 | Mean fit time: 0.018700 s
| epochs=100 | CV MSE=0.084016 pm 0.015041 | Mean fit time: 0.036189 s
| epochs=300 | CV MSE=0.018161 pm 0.003795 | Mean fit time: 0.110235 s <--------- 1SE
| epochs=1000 | CV MSE=0.018446 pm 0.003766 | Mean fit time: 0.373078 s
| epochs=2000 | CV MSE=0.017932 pm 0.004010 | Mean fit time: 0.747514 s <--------- MIN 
| epochs=3000 | CV MSE=0.018083 pm 0.003630 | Mean fit time: 1.120546 s





## (7)
# Final prediction on test data NN

--------------------------------------------------------------------------------
(No penalty) Test-MSE pm SE: 0.01802 pm 0.00579, [50, 1], parameters 151
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(L2) Test-MSE pm SE: 0.02006 pm 0.00807, [50, 1], parameters 151
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(L1) Test-MSE pm SE: 0.02080 pm 0.00845, [50, 1], parameters 151
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(No penalty) Test-MSE pm SE: 0.01285 pm 0.00466, [100, 100, 1], parameters 10401
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(L2) Test-MSE pm SE: 0.01782 pm 0.00631, [100, 100, 1], parameters 10401
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
(L1) Test-MSE pm SE: 0.01620 pm 0.00494, [100, 100, 1], parameters 10401
--------------------------------------------------------------------------------


## (8)
# Final prediction on test data regression methods OLS, Ridge and Lasso
--------------------------------------------------------------------------------
test MSE (OLS): 0.019 pm 0.005
--------------------------------------------------------------------------------
test MSE (Ridge): 0.018 pm 0.005
--------------------------------------------------------------------------------
test MSE (Lasso): 0.023 pm 0.009
--------------------------------------------------------------------------------



"""



