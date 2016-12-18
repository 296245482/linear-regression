function [w,b,cost] = Train(x,y,tol,max_iter)
    fprintf('training started\n');
    n = size(x,1);
    N = size(x,2);
    w = ones(n,1)/n;
    b = 1;
    cost = [];
    count = 1;
    while 1
        partial_w = zeros(n,1);
        partial_b = 0;
        %找一个梯度下降方向
        for i = 1:N
            a = exp(w'*x(:,i)+b);
            partial_w = partial_w + (1/(a+1)-y(i))*(-1)/((a+1)*(a+1))*a*x(:,i);
            partial_b = partial_b + (1/(a+1)-y(i))*(-1)/((a+1)*(a+1))*a;
        end
        
        %比较下降后的数值
        cost_temp = Cost(x,y,w,b);
        step = 1;
        while step > 1e-12
            w_temp = w - step*partial_w;
            b_temp = b - step*partial_b;
            new_cost = Cost(x,y,w_temp,b_temp);
            if new_cost < cost_temp
                break;
            end
            step = step * 0.1;
        end
        
        %结束循环的最小步长
        if step <= 1e-12
            fprintf('reach the min step size after %d iterates\n',count);
            break;
        end
        
        w = w_temp;
        b = b_temp;
        
        cost = [cost,new_cost];
        
        %结束循环的最小代价
        if new_cost < tol
            fprintf('reach the aimed cost after %d iterates\n',count);
            break;
        end
        
        %结束循环的最大递归数
        if count > max_iter
            fprintf('training finished after %d iterates,didn''t reach the aimed precision\n',count);
            break;
        end
        
        count = count+1;
    end
end