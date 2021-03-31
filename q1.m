%Load Features and labels 
X_Data = load('data.txt');
Y_Data = load('labels.txt');
%Replace 0 with -1 for negative samples 
Y_Data(Y_Data==0)=-1;
%Create train and test dataset
X_Test = X_Data(2001:4601,:);
Y_Test = Y_Data(2001:4601);
N_Test = size(X_Test,1);
%Create range of training samples to be used for training
N_trains = [200,500,800,1000,1500,2000];

maxiter = 1000;
epsilon = .0005;
err = 0 ;

%lrs = [.5,.05,.005,.0005,.00005];
lrs = [.0005];
lr_count = 0;
for lr = lrs
    lr_count = lr_count + 1 ;
Test_error = zeros(6,1);
test_count = 1;
for n_trains = N_trains
    X_train = X_Data(1:n_trains,:);
    Y_train = Y_Data(1:n_trains);
    %Get learned weights 
    weights = logistic_train(X_train,Y_train,epsilon,maxiter,lr);
    err = 0 ;
    %predict on test data
   for i = 1:N_Test
       sigm = 1 / (1 + exp(-(X_Test(i,:) * weights)));%predict labels based on model       
       if sigm > 0.5
            res = 1; %if the value is larger than 0.5, give a label of 1; otherwise,-1.
        else
            res = -1;
       end
       %calculate error
        if res ~= Y_Test(i)
            err = err + 1 ;
        end
   end
    Test_error(test_count) = 1 - err / N_Test ;%correct prediction rate
    test_count = test_count + 1 ;
end 
if lr_count == 2
hold on
end
plot(N_trains,Test_error);
end
hold off 
%lgd = legend('lr=.5','lr=.05','lr=.005','lr=.0005','lr=.00005');
lgd = legend('lr=.0005');
lgd.Location='south';
title('Testing Accuracy vs Train Samples for different learning rates');
xlabel('Training Samples');
ylabel('Test Accuracy');
grid on

function [weights] = logistic_train(data,labels,epsilon,maxiter,lr)
%Samples and Features size
[N,d] = size(data) ;
%Initialize weights 
weights_old = zeros(d,1);
weights = zeros(d,1);
%run only until max iteration
for i = 1:maxiter       
        loss_grad = zeros(d,1);%matrix with zeros
        %calculate gradient of loss functions for all train samples
        for k = 1:N
            loss_grad = loss_grad - (labels(k) * [data(k,:)]' *  (1 / (1 + exp(labels(k)*weights_old'*[data(k,:)]')))); 
        end
        %update weights using gradient descent
       weights = weights_old - lr*(loss_grad);
       %check absolute difference from previous weights
       avg_diff = sum(abs(weights-weights_old)) / d;
       weights_old = weights ;
       %check if lower than constraint
       if avg_diff < epsilon %stop if the difference between two iterations is smaller than tolerance
            break
       end
end    
end