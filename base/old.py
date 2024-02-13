def posteriorDistribution2(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from training set
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    N = x.shape[0]
    X = np.c_[np.ones(N), x]
    
    # Compute the covariance matrix and mean vector of the posterior distribution
    Sigma = np.linalg.inv(beta * X.T @ X + (1/sigma2) * np.eye(2))
    mu = (1/sigma2) * Sigma @ X.T @ z
    
    # Evaluate the density of the posterior distribution on a grid of values for a_0 and a_1
    a0_vals = np.linspace(-1, 1, 100)
    a1_vals = np.linspace(-1, 1, 100)
    a0_grid, a1_grid = np.meshgrid(a0_vals, a1_vals)
    a_grid = np.stack([a0_grid, a1_grid], axis=-1)
    posterior_density = util.density_Gaussian(mu, Sigma, a_grid).reshape(a0_grid.shape)
    
    # Plot the contours of the posterior density
    plt.contourf(a0_grid, a1_grid, posterior_density)
    plt.plot([-0.1], [-0.5], marker='o', markersize=4, color='red')
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('Posterior Distribution')
    plt.colorbar()
    plt.show()

    
    return (mu, Cov)

def posteriorDistribution3(x, z, beta, sigma2):

    alpha = 1
    # Define the prior distribution
    prior_mean = np.zeros(2)
    prior_cov = alpha * np.eye(2)

    # Define the likelihood function
    A = np.c_[np.ones_like(x), x]
    likelihood_cov = beta * np.eye(len(x))

    # Compute the posterior distribution
    Cov = np.linalg.inv(prior_cov + A.T.dot(np.linalg.inv(likelihood_cov)).dot(A))
    mu = Cov.dot(A.T).dot(np.linalg.inv(likelihood_cov)).dot(z)

    # Define the grid of values for plotting the posterior distribution
    a0_grid = np.linspace(-1, 1, 100)
    a1_grid = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0_grid, a1_grid)

    # Compute the posterior density over the grid
    samples = np.column_stack((A0.ravel(), A1.ravel()))
    density = np.array([util.density_Gaussian(mu, Cov, s) for s in samples])
    density = density.reshape(A0.shape)

    # Plot the contours of the posterior density
    plt.contour(A0, A1, density, colors='blue')
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='orange')
    plt.xlabel('a0')
    plt.ylabel('a1')
    if len(x) == 1:
        plt.title('posterior distribution based on 1 data sample')
        plt.savefig("posterior1.pdf")
    elif len(x) == 5:
        plt.title('posterior distribution based on 5 data samples')
        plt.savefig("posterior5.pdf")
    elif len(x) == 100:
        plt.title('posterior distribution based on 100 data samples')
        plt.savefig("posterior100.pdf")
    plt.show()
    
    return (mu, Cov)

def priorDistribution2(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    # Define the range of values for a0 and a1
    a0_range = np.linspace(-1, 1, 100)
    a1_range = np.linspace(-1, 1, 100)
    a0_grid, a1_grid = np.meshgrid(a0_range, a1_range)
    
    # Compute the prior density for each combination of a0 and a1
    prior_density = np.exp(-beta/2 * (a0_grid**2 + a1_grid**2))
    
    # Plot the contours of the prior density
    plt.contour(a0_grid, a1_grid, prior_density)
    plt.plot([-0.1], [-0.5], marker='o', markersize=4, color='red')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution p(a)')
    plt.show()
    
    return 