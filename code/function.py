import numpy as np;
import cv2;



def costFunction(params, Y, R, num_users, num_movies, num_features, lambda_):
    
    params = params.T;
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features));
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features));

    J = 0;
    X_grad = np.mat(np.zeros((X.shape[0], X.shape[1])));
    Theta_grad = np.mat(np.zeros((Theta.shape[0], Theta.shape[1])));

    t = np.power(np.dot(X, Theta.T)-Y, 2);
    J = np.sum(np.multiply(t, R))/2 + lambda_/2*np.sum(np.power(X, 2)) + lambda_/2*np.sum(np.power(Theta, 2));
    X_grad = np.dot(np.multiply(np.dot(X, Theta.T)-Y, R),Theta) + lambda_*X;
    Theta_grad = np.dot(np.multiply(np.dot(X, Theta.T)-Y, R).T,X) + lambda_*Theta;

    grad = np.mat(np.hstack((X_grad.reshape(num_movies*num_features), Theta_grad.reshape(num_users*num_features))));

    return J, grad;


def loadmovies(path):

    movie_list = [];

    with open(path, encoding='latin-1') as f:
        for line in f:
            tokens = line.strip().split(' ');
            movie_list.append(' '.join(tokens[1:]));

    movie_list = np.array(movie_list);

    return movie_list;

def normalize(Y, R):

    m = np.sum(R, axis=1);

    Y_sum = np.sum(Y, axis=1);

    Y_mean = Y_sum/m;

    Y_norm = Y - np.repeat(Y_mean, Y.shape[1], axis=1);

    return Y_norm, Y_mean;


