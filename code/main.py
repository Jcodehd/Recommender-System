import scipy.io as sc
import scipy.optimize as scio;
import numpy as np;
import function as f


# 加载数据
data = sc.loadmat('Machine Learning/Recommender System/ex8_movies.mat');
Y = np.mat(data['Y']); # 1682*943
R = np.mat(data['R']); # 1682*943
movie_list = f.loadmovies('Machine Learning/Recommender System/movie_ids.txt');
print(movie_list);

# 添加自己的训练样本
ratings = np.zeros(1682);
ratings[0] = 4;ratings[6] = 3;ratings[11] = 5;ratings[53] = 4;ratings[63] = 5;ratings[65] = 3;ratings[68] = 5;ratings[97] = 2;ratings[182] = 4;ratings[225] = 5;ratings[354] = 5;

# 插入到Y、R中
Y = np.insert(Y, 0, ratings, axis=1);
R = np.insert(R, 0, ratings!=0, axis=1);

#获取电影数、用户数、特征数
num_movies = Y.shape[0];
num_users = Y.shape[1];
num_features = 50;
lambda_ = 10;

# 初始化 X、 Theta
X = np.random.standard_normal((num_movies, num_features));
Theta = np.random.standard_normal((num_users, num_features));
params = np.mat(np.concatenate((X.ravel(), Theta.ravel())));

# 均值化
Y_norm, Y_mean = f.normalize(Y, R);

# 训练

result = scio.fmin_tnc(func=f.costFunction, x0=params, args=(Y_norm, R, num_users, num_movies, num_features, lambda_));

X_ = np.reshape(result[0][:num_movies*num_features], (num_movies, num_features));
Theta_ = np.reshape(result[0][num_movies*num_features:], (num_users, num_features));

p = np.mat(np.dot(X_, Theta_.T)[: , 0]).T + Y_mean;

idx = np.argsort(p, axis=0)[::-1];

for m in movie_list[idx][:10]:
    print(m)
























