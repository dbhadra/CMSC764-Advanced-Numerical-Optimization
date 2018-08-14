%% Q1(b) to test gradient checker. 
s = rng;
rng(s)
x = sym('x',[3 5]);
A = randn(4,3);
b = randn(4,5);
f = @(x) 0.5*(norm(A*x-b,'fro'))^2;
x0 = randn(3,5);
grad = @(x) A'*(A*x-b);

output = check_gradient(f,grad,x0)

%% Q2(a) to product gradient of logistic regression function


%% Q2(b) to test gradient with classification problem
[ D,c ] = create_classification_problem(1000, 10, 5);  
x = randn(10,1);
f = @logreg_objective; 
grad = @logreg_grad;
x = {x D c};
output = check_gradient(f,grad,x)

%% Q3(e) to test gradient using noisy test image
b = phantom(64);
mu=0.5;
x = sym('x',[64 64]);
f1 = @tv_objective;
fprintf('The objective f(x) is');
f2 = @tv_grad;
fprintf('The gradient of f(x) is');
x0 = randn(size(b,1),size(b,2));
x0 = {x0 b mu eps};
output = check_gradient(f1,f2,x0)

%% Q4(a) to implement the gradient of a neural network 



%% Q4(b) to verify gradient for a neural network 
load_mnist;
W{1} = randn(784,20); % Random Gaussian Weights
W{2} = randn(20,15);
W{3} = randn(15,10);
vec = cell_to_vec(W);
D = x_train;
L = y_train; % Creating a one-hot label representation
f1 = @net_objective; 
fprintf('The loss of the neural network is')
feval(f1,W,D,L)
f2 = @net_grad;





