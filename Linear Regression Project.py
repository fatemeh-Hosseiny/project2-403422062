
from google.colab import files
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import pandas as pd # Import pandas for potentially nicer table output

# --- Helper Functions ---

def polynomial_basis_function(x, degree):
    """
    Polynomial basis function.
    Args:
        x (np.ndarray): Input feature(s). Must be scaled if using high degrees.
        degree (int): Polynomial degree.
    Returns:
        np.ndarray: Matrix of basis function outputs (Phi).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1) # Ensure x is a column vector
    # For multi-variate polynomials, this function needs to be extended
    # This simple version is for uni-variate input
    Phi = np.ones((x.shape[0], degree + 1))
    for i in range(1, degree + 1):
        Phi[:, i] = x[:, 0] ** i
    return Phi

def calculate_rmsd(y_true, y_pred):
    """
    Calculates Root Mean Square Deviation (RMSD).
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
    Returns:
        float: RMSD value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Learning Methods Implementation ---

def closed_form_ridge(Phi, y, lambda_reg):
    """
    Trains a linear regression model using Closed Form with Ridge regularization.
    Args:
        Phi (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        lambda_reg (float): Regularization coefficient (lambda).
    Returns:
        np.ndarray: Learned weights (w).
    """
    # Equation: w = (Phi^T * Phi + lambda * I)^(-1) * Phi^T * y
    Phi_T = Phi.T
    Identity = np.eye(Phi.shape[1])
    try:
        # Add a small value to the diagonal for numerical stability in inversion
        # (Phi_T @ Phi + lambda_reg * Identity)
        # Use a slightly larger value if still facing numerical issues, but lambda_reg should handle it
        w = np.linalg.solve(Phi_T @ Phi + lambda_reg * Identity, Phi_T @ y)
        # w = np.linalg.inv(Phi_T @ Phi + lambda_reg * Identity) @ Phi_T @ y # Alternative using inverse, solve is often more stable
    except np.linalg.LinAlgError:
        print(f"Warning: Could not solve closed form equation for lambda={lambda_reg}. Matrix might be singular.")
        return None
    return w

def gradient_descent(Phi, y, lambda_reg, regularization_type='ridge', learning_rate=0.01, n_iterations=1000):
    """
    Trains a linear regression model using Gradient Descent.
    Args:
        Phi (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        lambda_reg (float): Regularization coefficient (lambda).
        regularization_type (str): 'ridge' or 'lasso'.
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of training iterations.
    Returns:
        np.ndarray: Learned weights (w).
    """
    n_samples, n_features = Phi.shape
    w = np.zeros(n_features) # Initialize weights

    for i in range(n_iterations):
        # Calculate predictions
        y_pred = Phi @ w

        # Calculate error
        error = y_pred - y

        # Calculate gradient of the data fitting term
        gradient_data = (Phi.T @ error) / n_samples

        # Calculate gradient of the regularization term
        if regularization_type == 'ridge':
            gradient_reg = lambda_reg * w
        elif regularization_type == 'lasso':
            # Subgradient of L1 norm is the sign function
            # Add a small epsilon to avoid sign(0) issues if needed, but np.sign handles 0 as 0.
            gradient_reg = lambda_reg * np.sign(w)
        else:
            raise ValueError("Invalid regularization type. Use 'ridge' or 'lasso'.")

        # Calculate total gradient
        total_gradient = gradient_data + gradient_reg

        # Update weights
        w -= learning_rate * total_gradient

        # Check for NaN/Inf in weights - early stopping if divergence occurs
        if not np.all(np.isfinite(w)):
             print(f"Warning: GD diverged with lambda={lambda_reg}, method={regularization_type}. Weights became NaN/Inf.")
             return None # Return None to indicate training failure

    return w

def stochastic_gradient_descent(Phi, y, lambda_reg, regularization_type='ridge', learning_rate=0.01, n_iterations=1000):
    """
    Trains a linear regression model using Stochastic Gradient Descent.
    Args:
        Phi (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        lambda_reg (float): Regularization coefficient (lambda).
        regularization_type (str): 'ridge' or 'lasso'.
        learning_rate (float): Learning rate for stochastic gradient descent.
        n_iterations (int): Number of training iterations (epochs).
    Returns:
        np.ndarray: Learned weights (w).
    """
    n_samples, n_features = Phi.shape
    w = np.zeros(n_features) # Initialize weights

    for i in range(n_iterations):
        # Shuffle data for each epoch (optional but recommended)
        permutation = np.random.permutation(n_samples)
        Phi_shuffled = Phi[permutation]
        y_shuffled = y[permutation]

        for j in range(n_samples):
            # Get one sample
            phi_j = Phi_shuffled[j].reshape(1, -1) # Reshape to (1, n_features)
            y_j = y_shuffled[j]

            # Calculate prediction for the sample
            y_pred_j = phi_j @ w

            # Calculate error for the sample
            error_j = y_pred_j - y_j

            # Calculate gradient of the data fitting term for the sample
            gradient_data_j = phi_j.T * error_j # Shape (n_features, 1)

            # Calculate gradient of the regularization term
            if regularization_type == 'ridge':
                gradient_reg = lambda_reg * w
            elif regularization_type == 'lasso':
                # Subgradient of L1 norm is the sign function
                # Add a small epsilon to avoid sign(0) issues if needed, but np.sign handles 0 as 0.
                gradient_reg = lambda_reg * np.sign(w)
            else:
                 raise ValueError("Invalid regularization type. Use 'ridge' or 'lasso'.")

            # Calculate total gradient (gradient_data_j is for one sample, gradient_reg is for the whole model)
            # Note: The scaling of gradient_data_j varies in SGD implementations.
            # Here, we don't divide by n_samples inside the inner loop.
            total_gradient = gradient_data_j.flatten() + gradient_reg # Flatten gradient_data_j

            # Update weights
            w -= learning_rate * total_gradient

             # Check for NaN/Inf in weights - early stopping if divergence occurs
            if not np.all(np.isfinite(w)):
                 print(f"Warning: SGD diverged with lambda={lambda_reg}, method={regularization_type}. Weights became NaN/Inf.")
                 return None # Return None to indicate training failure


    return w

# --- Main Execution ---

# 1. Generate or Load Data
# Using synthetic data for demonstration
np.random.seed(0)
n_samples = 100
# Generate a simple non-linear relationship with noise
X = np.random.rand(n_samples, 1) * 10
y_true = np.sin(X[:, 0]) * 2 + X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.8 # Example non-linear data

# 5. Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# --- Feature Scaling ---
# Scale the input data X BEFORE applying polynomial basis functions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.transform(X) # Scale the full dataset for potential plotting later

# 2. Choose a family of basis functions and Enrich Hypothesis Space
# Using polynomial basis with a relatively high degree to allow interpolation/overfitting
polynomial_degree = 15 # High degree to potentially cause overfitting
# Now apply polynomial basis to the SCALED data
Phi_train = polynomial_basis_function(X_train_scaled, polynomial_degree)
Phi_test = polynomial_basis_function(X_test_scaled, polynomial_degree)
Phi_full = polynomial_basis_function(X_full_scaled, polynomial_degree) # For plotting the true function if needed

# Define methods and regularization types
methods = {
    'CF-Ridge': {'type': 'cf', 'reg': 'ridge'},
    'GD-Ridge': {'type': 'gd', 'reg': 'ridge'},
    'GD-Lasso': {'type': 'gd', 'reg': 'lasso'},
    'SGD-Ridge': {'type': 'sgd', 'reg': 'ridge'},
    'SGD-Lasso': {'type': 'sgd', 'reg': 'lasso'}
}

# 4. Set regularization coefficient values (lambda)
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] # Log-scale values

# Store results
results_rmsd_test = {}
results_rmsd_train = {}
results_nonzero_weights = {} # For Lasso methods

# --- Training and Evaluation Loop ---

print("Training models and evaluating performance...")
for method_name, method_params in methods.items():
    results_rmsd_test[method_name] = []
    results_rmsd_train[method_name] = []
    if method_params['reg'] == 'lasso':
         results_nonzero_weights[method_name] = []
    else:
         # Store None or empty list for non-Lasso methods
         results_nonzero_weights[method_name] = [None] * len(lambda_values)


    print(f"\n--- Method: {method_name} ---") # Added separation for clarity
    for lambda_reg in lambda_values:
        print(f"  Lambda: {lambda_reg}")
        w = None # Initialize weights variable

        # 3. Train the model
        if method_params['type'] == 'cf':
            w = closed_form_ridge(Phi_train, y_train, lambda_reg)
        elif method_params['type'] == 'gd':
            # Adjusted learning rate for stability. May need further tuning.
            # Increased iterations as LR is smaller.
            w = gradient_descent(Phi_train, y_train, lambda_reg,
                                 regularization_type=method_params['reg'],
                                 learning_rate=0.0001, n_iterations=20000) # Slightly increased iterations
        elif method_params['type'] == 'sgd':
            # Adjusted learning rate for stability. May need further tuning.
            # Increased iterations as LR is smaller. SGD often needs more.
             w = stochastic_gradient_descent(Phi_train, y_train, lambda_reg,
                                            regularization_type=method_params['reg'],
                                            learning_rate=0.00001, n_iterations=50000) # Significantly increased iterations


        if w is not None: # Check if training was successful (didn't return None)
            # Predict on training and test sets
            y_train_pred = Phi_train @ w
            y_test_pred = Phi_test @ w

            # Calculate RMSD
            rmsd_train = calculate_rmsd(y_train, y_train_pred)
            rmsd_test = calculate_rmsd(y_test, y_test_pred)

            results_rmsd_train[method_name].append(rmsd_train)
            results_rmsd_test[method_name].append(rmsd_test)

            # 6. Compare and explain the number of non-zero weights (for Lasso)
            if method_params['reg'] == 'lasso':
                # Count non-zero weights (using a tolerance for floating point numbers)
                # A smaller tolerance like 1e-12 might be needed depending on precision
                nonzero_count = np.sum(np.abs(w) > 1e-9) # Threshold for non-zero
                results_nonzero_weights[method_name].append(nonzero_count)
            # else: Already initialized with None

        else:
             # Handle cases where training failed (e.g., singular matrix in CF or divergence in GD/SGD)
             results_rmsd_train[method_name].append(np.nan)
             results_rmsd_test[method_name].append(np.nan)
             if method_params['reg'] == 'lasso':
                 results_nonzero_weights[method_name].append(np.nan)


# --- 5. Summarize Results in Tables ---

print("\n" + "="*60) # Separator for results
print("--- Summary of Results ---")
print("="*60)

print("\n--- RMSD on Test Set ---")
# Using pandas DataFrame for nicer table output
rmsd_test_df = pd.DataFrame(results_rmsd_test, index=[f"Lambda={l}" for l in lambda_values]).T
print(rmsd_test_df.to_string(float_format='{:.4f}'.format)) # Use to_string for full output in Colab

print("\n--- RMSD on Training Set ---")
rmsd_train_df = pd.DataFrame(results_rmsd_train, index=[f"Lambda={l}" for l in lambda_values]).T
print(rmsd_train_df.to_string(float_format='{:.4f}'.format)) # Use to_string for full output in Colab


print("\n--- Number of Non-zero Weights (Lasso Methods) ---")
# Filter out non-Lasso methods and create DataFrame
lasso_nonzero_results = {method: results_nonzero_weights[method] for method in ['GD-Lasso', 'SGD-Lasso'] if method in results_nonzero_weights}
nonzero_weights_df = pd.DataFrame(lasso_nonzero_results, index=[f"Lambda={l}" for l in lambda_values]).T
print(nonzero_weights_df.to_string()) # Use to_string for full output in Colab


# --- Plotting ---

print("\n" + "="*60) # Separator for plots
print("--- Generating Plots ---")
print("="*60)

plt.figure(figsize=(12, 6))
for method_name, rmsds in results_rmsd_test.items():
    # Filter out None/NaN values for plotting if any method failed
    valid_lambdas = [lambda_values[i] for i, rmsd in enumerate(rmsds) if not np.isnan(rmsd)]
    valid_rmsds = [rmsd for rmsd in rmsds if not np.isnan(rmsd)]
    if valid_lambdas: # Only plot if there are valid points
        plt.plot(valid_lambdas, valid_rmsds, marker='o', linestyle='-', label=method_name)

plt.xscale('log') # Use log scale for lambda
plt.xlabel("Regularization Coefficient (Lambda)")
plt.ylabel("RMSD on Test Set")
plt.title("RMSD vs. Lambda for Different Learning Methods")
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()

# Plot for non-zero weights for Lasso
plt.figure(figsize=(12, 6))
for method_name in ['GD-Lasso', 'SGD-Lasso']:
     if method_name in results_nonzero_weights:
         valid_lambdas_nz = [lambda_values[i] for i, count in enumerate(results_nonzero_weights[method_name]) if count is not None and not np.isnan(count)]
         valid_counts_nz = [count for count in results_nonzero_weights[method_name] if count is not None and not np.isnan(count)]
         if valid_lambdas_nz: # Only plot if there are valid points
            plt.plot(valid_lambdas_nz, valid_counts_nz, marker='o', linestyle='-', label=method_name)

plt.xscale('log') # Use log scale for lambda
plt.xlabel("Regularization Coefficient (Lambda)")
plt.ylabel("Number of Non-zero Weights")
plt.title("Number of Non-zero Weights vs. Lambda for Lasso Methods")
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()


# --- Conclusion and Analysis ---
print("\n" + "="*60) # Separator for conclusion
print("--- Conclusion and Analysis ---")
print("="*60)

print("""
Based on the results from the tables and plots above, we can draw several conclusions:

1.  **Effect of Regularization (Lambda) on RMSD:**
    Observe the 'RMSD on Test Set' table and plot. As lambda increases, the RMSD on the test set typically decreases initially, reaching a minimum, and then increases again for larger lambda values. This demonstrates how appropriate regularization helps prevent overfitting (reducing test error) but excessive regularization can lead to underfitting (increasing test error).
    -   *Analyze:* Which lambda value gives the minimum RMSD on the test set for each method? How does this optimal lambda compare across methods?

2.  **Comparison of Learning Methods (CF, GD, SGD):**
    Compare the RMSD values obtained by Closed Form (CF), Gradient Descent (GD), and Stochastic Gradient Descent (SGD) for the same lambda values.
    -   *Analyze:* Do GD and SGD achieve similar performance to CF for appropriate lambda? Are there differences in performance? Note that GD/SGD performance heavily depends on the learning rate and number of iterations. If they didn't converge fully, their RMSD might be higher than CF.

3.  **Comparison of Regularization Types (Ridge vs. Lasso):**
    Compare the RMSD values for GD-Ridge vs. GD-Lasso and SGD-Ridge vs. SGD-Lasso.
    -   *Analyze:* Does one type of regularization consistently perform better than the other in terms of RMSD on the test set?

4.  **Effect of Lasso Regularization on Weights:**
    Examine the 'Number of Non-zero Weights' table and plot for Lasso methods. As lambda increases, the number of non-zero weights should decrease.
    -   *Analyze:* At what lambda value do weights start becoming zero? Does a larger lambda result in a sparser model (more zero weights)? How does this relate to feature selection? Discuss any observations about which features (corresponding to which weight indices) tend to become zero first.

5.  **Overall Best Performing Method and Lambda:**
    Based on the RMSD on the test set, identify which combination of learning method and lambda value seems to provide the best generalization performance for this dataset and chosen basis functions.

**Further Discussion:**
Consider the trade-off between model complexity (degree of polynomial basis), training error (RMSD on training set), and generalization error (RMSD on test set). How does regularization help manage this trade-off?
If you used a real-world dataset, discuss the implications of feature selection by Lasso – do the features that get zero weights make sense in the context of the problem?

---
""")

plt.grid(True, which="both", linestyle='--')
plt.savefig('rmsd_vs_lambda.png') # Save the RMSD plot
plt.show() # Display the plot in Colab

# Plot for non-zero weights for Lasso vs. Lambda
plt.figure(figsize=(12, 6))
for method_name in ['GD-Lasso', 'SGD-Lasso']:
     if method_name in results_nonzero_weights:
         valid_lambdas_nz = [lambda_values[i] for i, count in enumerate(results_nonzero_weights[method_name]) if count is not None and not np.isnan(count)]
         valid_counts_nz = [count for count in results_nonzero_weights[method_name] if count is not None and not np.isnan(count)]
         if valid_lambdas_nz:
            plt.plot(valid_lambdas_nz, valid_counts_nz, marker='o', linestyle='-', label=method_name)

plt.xscale('log')
plt.xlabel("Regularization Coefficient (Lambda)")
plt.ylabel("Number of Non-zero Weights")
plt.title("Number of Non-zero Weights vs. Lambda for Lasso Methods")
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.savefig('nonzero_weights_vs_lambda.png') # Save the non-zero weights plot
plt.show() # Display the plot in Colab
