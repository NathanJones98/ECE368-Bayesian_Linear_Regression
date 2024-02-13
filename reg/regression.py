import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    # Define the mean vector and covariance matrix for the prior
    mean_vec = np.zeros(2)
    cov_mat = (beta) * np.eye(2) # p (a) = N([0; 0], [B 0; 0 B] ) 

    # Generate a grid of values for a0 and a1
    a0, a1 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    a = np.stack([a0, a1], axis=-1).reshape(-1, 2)
    
    # Evaluate the density of the prior at each point on the grid
    prior_density = util.density_Gaussian(mean_vec, cov_mat, a).reshape(a0.shape)
    
    # Plot the contours of the prior density
    plt.contourf(a0, a1, prior_density)

    plt.grid()
    plt.plot([-0.1], [-0.5], marker='o', markersize=4, color='red')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution P(a)')
    plt.savefig("prior.pdf")
    plt.show()
    return

def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    cov_w_i = 1/sigma2
    cov_mat_i = (1/beta) * np.eye(2) 

    a0, a1 = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    a = np.stack([a0, a1], axis=-1).reshape(-1, 2)
    A = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    mu = np.matmul(np.linalg.inv((cov_mat_i + cov_w_i * np.matmul(A.T, A))), cov_w_i * np.matmul(A.T, z))
    Cov = np.linalg.inv((cov_mat_i + cov_w_i * np.matmul(A.T, A)))

    posterior = util.density_Gaussian(mu.T, Cov, a).reshape(a0.shape)

    # plot the contours
    plt.grid()
    plt.contourf(a0, a1, posterior)
    plt.plot([-0.1], [-0.5], marker='o', markersize=4, color='red')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title("Posterior [p(a|x1, z1, . . . , xN , zN )] for N=" + f'{x.shape[0]}')
    plt.savefig("posterior" + f'{x.shape[0]}' + ".pdf")
    plt.show()

    return (mu, Cov)

def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 

    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot

    Outputs: None
    -----
    """
    A = np.append(np.ones((len(x), 1)), np.expand_dims(x, 1), axis=1)

    mu_reshaped = np.reshape(mu, (2, 1)).squeeze()
    muz = np.matmul(A, mu_reshaped)
    Covz = np.matmul(np.matmul(A, Cov), A.T) + sigma2
    dev = np.sqrt(np.diag(Covz))

    plt.grid()
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.scatter(x_train, z_train)
    plt.errorbar(x, muz, yerr=dev, fmt='rx')
    plt.title("predict [p(z|x, x1, z1, . . . , xN , zN )] for N=" + f'{x_train.shape[0]}')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.plot([-0.1], [-0.5], marker='o', markersize=4, color='red')
    plt.savefig("predict" + f'{x_train.shape[0]}' + ".pdf")
    plt.show()

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # prior distribution p(a)
    priorDistribution(beta)

    # number of training samples used to compute posterior 
    for ns in [1, 5, 100]:
        
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        #print(mu, Cov)
        #input("Pause")
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
        

   

    
    
    

    
