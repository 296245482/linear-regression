function cost = Cost(x,y,w,b)
cost = 0;
N = size(x,2);
%平方代价
for i = 1:N
    temp_cost = (1/(1+exp(w'*x(:,i)+b))-y(i));
    cost = cost + temp_cost*temp_cost;
end

