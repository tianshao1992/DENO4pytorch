%  Use double precision to generate data (due to GP sampling)

N = 512;
% gp 参数尺度
gp_scale = 1.0;  % 大小
gp_length = 0.05; % 尺度，用于控制函数的复杂度，越大越平滑
x_min = 0.;
x_max = 1.;

jitter = 1e-10;
X = linspace(x_min, x_max, N)';
K = RBF1d(X, X', gp_scale, gp_length);
L = chol(K + jitter * eye(N));

% 生成M个不同函数
M = 30;
for i = 1:M
%     生成 
    rng(i);
    gp_sample = L' * randn(N, 1);
    plot(gp_sample);
    hold on;
end


function [K] = RBF1d(x1, x2, scale, length)

diffs = (x1 - x2) / length;
r2 = diffs .^ 2;

K = scale * exp(-0.5 * r2);
end 

