import numpy as np
import matplotlib.pyplot as plt

def density_Gaussian(mean_vec,covariance_mat,x_set):
    """ Return the density of multivariate Gaussian distribution
        Inputs: 
            mean_vec is a 1D array (like array([,,,]))
            covariance_mat is a 2D array (like array([[,],[,]]))
            x_set is a 2D array, each row is a sample
        Output:
            a 1D array, probability density evaluated at the samples in x_set.
    """
    d = x_set.shape[1]  
    inv_Sigma = np.linalg.inv(covariance_mat)
    det_Sigma = np.linalg.det(covariance_mat)
    density = []
    for x in x_set:
        x_minus_mu = x - mean_vec
        exponent = - 0.5*np.dot(np.dot(x_minus_mu,inv_Sigma),x_minus_mu.T)
        prob = 1/(((2*np.pi) ** (d/2))*np.sqrt(det_Sigma))*np.exp(exponent)
        density.append(prob)
    density_array = np.array(density)  
    
    return density_array 

def get_data_in_file(filename):
    """ 
    Read the input/traget data from the given file as arrays 
    """
    with open(filename, 'r') as f:
        data = []
        # read the data line by line
        for line in f: 
            data.append([float(x) for x in line.split()]) 
            
    # store the inputs in x and the tragets in z      
    data_array = np.array(data)     
    x = data_array[:,0:1]   # 2D array
    z = data_array[:,1:2]   # 2D array
    
    return (x, z)
    
    

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
    a0g = np.arange(-1, 1, 0.01);
    a1g = np.arange(-1, 1, 0.01);
    a0, a1 = np.meshgrid(a0g, a1g);
    
    a0f = a0.flatten().reshape(-1, 1);
    a1f = a1.flatten().reshape(-1, 1);
    
    input = np.concatenate((a0f, a1f), axis = 1);
    contour = density_Gaussian([0, 0], [[beta, 0], [0, beta]], input).reshape((200, 200));
        
    plt.contour(a0, a1, contour);
    plt.plot([-0.1], [-0.5], marker='o');
    
    plt.grid();
    plt.title("p(a)");
    plt.xlabel("a0");
    plt.ylabel("a1");
    plt.savefig('prior.pdf')
    plt.show();
    
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
    a0g = np.arange(-1, 1, 0.01);
    a1g = np.arange(-1, 1, 0.01);
    a0, a1 = np.meshgrid(a0g, a1g);
    
    a0f = a0.flatten().reshape(-1, 1);
    a1f = a1.flatten().reshape(-1, 1);
    
    A = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1);
    inv_cov_a = np.array([[1 / beta, 0], [0, 1 / beta]]);

    mu = np.matmul(np.linalg.inv(inv_cov_a + np.matmul(A.T, A) / sigma2), np.matmul(A.T, z) / sigma2);
    Cov = np.linalg.inv(inv_cov_a + np.matmul(A.T, A) / sigma2);

    input = np.concatenate((a0f, a1f), axis = 1);
    contour = density_Gaussian(mu.T[0], Cov, input).reshape((200, 200));
        
    plt.contour(a0, a1, contour);
    plt.plot([-0.1], [-0.5], marker='o');
    
    plt.grid();
    plt.title("p(a|x, z) for " + str(x.shape[0]) + " Training Samples");
    plt.xlabel("a0");
    plt.ylabel("a1");
    plt.savefig("posterior" + str(x.shape[0]) + ".pdf")
    plt.show();
   
    return (mu,Cov)


def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    A = np.append(np.ones((len(x), 1)), np.expand_dims(x, 1), axis=1);
    mu_z = np.matmul(A, mu);
    cov_z = np.matmul(np.matmul(A, Cov), A.T) + sigma2;
    stddev_z = np.sqrt(np.diag(cov_z));

    plt.xlim([-4, 4]);
    plt.ylim([-4, 4]);
    plt.xlabel('x');
    plt.ylabel('z');
    plt.grid();
    plt.title("p(z|x, z) for " + str(x_train.shape[0]) + " Training Samples");
    plt.scatter(x_train, z_train);
    plt.errorbar(x, mu_z, yerr=stddev_z, fmt='rx');
    plt.legend(["Training Samples", "Predicted Targets Within 1 Standard Deviation"], loc='best');
    plt.savefig("predict" + str(x_train.shape[0]) + ".pdf")
    plt.show();
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = get_data_in_file('/content/gdrive/MyDrive/Colab Notebooks/368 Lab2/training.txt')
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
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)