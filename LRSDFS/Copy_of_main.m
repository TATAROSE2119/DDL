% 获取当前工作目录
currentDir = pwd;

% 创建保存结果的文件夹，确保在当前项目目录下
if ~exist(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/convergence_error'), 'dir')
    mkdir(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/convergence_error'));
end
if ~exist(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/statistics'), 'dir')
    mkdir(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/statistics'));
end
if ~exist(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/dictionary'), 'dir')
    mkdir(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/plts/dictionary'));
end
if ~exist(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/convergence_error'), 'dir')
    mkdir(fullfile(currentDir, 'Documents/MATLAB/LRSDFS/convergence_error'));
end

% 初始化参数
k1 = 10; % 低秩
k2 = 21; % 稀疏
max_iter = 500;  % 最大迭代次数
tol = 1e-3;     % 收敛阈值
train_file = 'TE_data/test_data/d00_te.dat';
train_data = load(train_file);
%train_data = train_data(:,[1:22,42:52]);   % 取33个变量
train_data = train_data';
[n_samples, n_features] = size(train_data');
train_data=normalize(train_data,1);



% 初始化结果存储
results = [];


% 初始化矩阵
    W1 = rand(n_samples, k1);
    W2 = rand(n_samples, k2);
    Y1 = rand(k1, n_samples);
    Y2 = rand(k2, n_samples);
    U1 = rand(k1, k1);
    U2 = rand(k2, k2);
    i_d = ones(n_features, 1);
    i_c = ones(n_samples, 1);
    
    W1_norms_e = [];
    W2_norms_e = [];
    Y1_norms_e = [];
    Y2_norms_e = [];
    
    % 迭代更新
    for iteration = 1:max_iter
        XW_1 = train_data * W1;
        [M, S1, N] = svd(XW_1, 'econ');
        
        prev_W1 = W1;
        prev_W2 = W2;
        prev_Y1 = Y1;
        prev_Y2 = Y2;
        
        % 调用 MATLAB 版的函数更新 W1, W2, Y1, Y2, U1, U2
        W1 = update_W1(train_data, W1, W2, Y1, Y2, M, N, 0.3);
        W2 = update_W2(train_data, W1, W2, Y1, Y2, i_d, i_c, 0.1);
        Y1 = update_Y1(train_data, W1, W2, Y1, Y2, U1, 1e5, 10);
        Y2 = update_Y2(train_data, W1, W2, Y1, Y2, U2, i_d, i_c, 0.3, 1e5, 10);
        [U1, U2] = update_U(Y1, Y2);
        
        % 计算收敛误差
        W1_norms_e = [W1_norms_e; norm(W1 - prev_W1)];
        W2_norms_e = [W2_norms_e; norm(W2 - prev_W2)];
        Y1_norms_e = [Y1_norms_e; norm(Y1 - prev_Y1)];
        Y2_norms_e = [Y2_norms_e; norm(Y2 - prev_Y2)];
        
        if calculate_relerr(W1, prev_W1, Y1, prev_Y1, W2, prev_W2, Y2, prev_Y2) < tol
            disp('Convergence reached.');
            break;
        end
    end
    
    % 计算字典矩阵
    D1 = train_data * W1;
    D2 = train_data * W2;
    


% 处理数据 d00 到 d21
for i = 1:21
    test_file = sprintf('TE_data/test_data/d%02d_te.dat', i);
    fprintf("Processing Train Data: %s, Test Data: %s\n", train_file, test_file);
    
    test_data = load(test_file);
    %test_data = test_data(:,[1:22,42:52]);
    if size(test_data,1) < size(test_data,2)
        test_data = test_data';
    end
    test_data = test_data';
    test_data=normalize(test_data,1);
    
    
    % 计算统计量
    [T2_statistics, SPE_statistics] = calculate_statistics(test_data, D1, D2, 0.9,0.1 );
    [T2_train, SPE_train] = calculate_statistics(train_data, D1, D2, 0.9,0.1);
    
    % 计算控制限
    alpha = 0.99;
    T2_limit = prctile(T2_train, alpha * 100);
    SPE_limit = prctile(SPE_train, alpha * 100);
    
    % 计算FDR和FAR
    T2_FDR = mean(T2_statistics(161:960) >= T2_limit);
    SPE_FDR = mean(SPE_statistics(161:960) >= SPE_limit);
    T2_FAR = mean(T2_statistics(1:160) >= T2_limit);
    SPE_FAR = mean(SPE_statistics(1:160) >= SPE_limit);
    
    fprintf('T2 FDR for d%02d: %.4f\n', i, T2_FDR);
    fprintf('SPE FDR for d%02d: %.4f\n', i, SPE_FDR);
    fprintf('T2 FAR for d%02d: %.4f\n', i, T2_FAR);
    fprintf('SPE FAR for d%02d: %.4f\n', i, SPE_FAR);
    
    results = [results; {sprintf('d%02d', i), T2_FDR, SPE_FDR, T2_FAR, SPE_FAR}];
    
    % 绘图并保存（不展示图形窗口）
    fig = figure('Visible', 'off'); % 创建不可见的图形窗口
    subplot(2,1,1);
    plot(T2_statistics);
    hold on;
    yline(T2_limit, 'r--');
    title(sprintf('T2 Statistics d%02d', i));

    subplot(2,1,2);
    plot(SPE_statistics);
    hold on;
    yline(SPE_limit, 'r--');
    title(sprintf('SPE Statistics d%02d', i));
    
    % 确保保存路径存在
    outputDir = 'Documents/MATLAB/LRSDFS/plts/statistics';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % 保存为 SVG 文件
    saveas(fig, sprintf('Documents/MATLAB/LRSDFS/plts/statistics/d%02d.svg', i));
    
    % 关闭图形窗口（可选，释放资源）
    close(fig);
end

% 保存结果到 CSV
fid = fopen('Documents/MATLAB/LRSDFS/all_results.csv', 'w');
fprintf(fid, 'file,T2_FDR,SPE_FDR,T2_FAR,SPE_FAR\n');

% 写入原始数据
for i = 1:size(results, 1)
    fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f\n', results{i, :});
end

% 计算每列的平均值（忽略第一列，因为它是字符串）
numericData = cell2mat(results(:, 2:end)); % 提取数值列并转换为矩阵
columnMeans = mean(numericData, 1); % 计算每列平均值
% 打印平均值到命令窗口
disp('每列的平均值：');
fprintf('T2_FDR: %.4f\n', columnMeans(1));
fprintf('SPE_FDR: %.4f\n', columnMeans(2));
fprintf('T2_FAR: %.4f\n', columnMeans(3));
fprintf('SPE_FAR: %.4f\n', columnMeans(4));

% 写入平均值行
fprintf(fid, 'Average,%.4f,%.4f,%.4f,%.4f\n', columnMeans(1), columnMeans(2), columnMeans(3), columnMeans(4));
% 关闭文件
fclose(fid);

disp('结果已保存到 Documents/MATLAB/LRSDFS/all_results.csv');


%%
function W1 = update_W1(X, W1, W2, Y1, Y2, M, N, a)
    numerator = X' * X * Y1';
    term1 = X' * X * W1 * Y1 * Y1';
    term2 = X' * X * W2 * Y2 * Y1';
    term3 = a * X' * M * N';
    denominator = term1 + term2 + term3;
    W1(denominator > 1e-8) = W1(denominator > 1e-8) .* numerator(denominator > 1e-8) ./ denominator(denominator > 1e-8);
end

function W2 = update_W2(X, W1, W2, Y1, Y2, i_d, i_c, b)
    numerator = X' * X * Y2';
    term1 = X' * X * W1 * Y1 * Y2';
    term2 = X' * X * W2 * Y2 * Y2';
    term3 = b * X' * (i_d * i_c') * Y2';
    denominator = term1 + term2 + term3;
    W2(denominator > 1e-8) = W2(denominator > 1e-8) .* numerator(denominator > 1e-8) ./ denominator(denominator > 1e-8);
end

function Y1 = update_Y1(X, W1, W2, Y1, Y2, U1, c, e)
    numerator = W1' * X' * X + 2 * e * Y1;
    term1 = W1' * X' * X * W1 * Y1;
    term2 = W1' * X' * X * W2 * Y2;
    term3 = c * U1 * Y1;
    term4 = 2 * e * Y1 * Y1' * Y1;
    denominator = term1 + term2 + term3 + term4;
    Y1(denominator > 1e-8) = Y1(denominator > 1e-8) .* numerator(denominator > 1e-8) ./ denominator(denominator > 1e-8);
end

function Y2 = update_Y2(X, W1, W2, Y1, Y2, U2, i_d, i_c, b, d, f)
    numerator = W2' * X' * X + 2 * f * Y2;
    term1 = W2' * X' * X * W1 * Y1;
    term2 = W2' * X' * X * W2 * Y2;
    term3 = b * W2' * X' * (i_d * i_c');
    term4 = d * U2 * Y2;
    term5 = 2 * f * Y2 * Y2' * Y2;
    denominator = term1 + term2 + term3 + term4 + term5;
    Y2(denominator > 1e-8) = Y2(denominator > 1e-8) .* numerator(denominator > 1e-8) ./ denominator(denominator > 1e-8);
end

function [U1, U2] = update_U(Y1, Y2)
    U1 = diag(1 ./ (2 * vecnorm(Y1, 2, 2)));
    U2 = diag(1 ./ (2 * vecnorm(Y2, 2, 2)));
    U1(isinf(U1)) = 0;
    U2(isinf(U2)) = 0;
end

function [T2_stats, SPE_stats] = calculate_statistics(X_new, D1, D2, a, b)
    n_samples = size(X_new, 2);
    T2_stats = zeros(n_samples, 1);
    SPE_stats = zeros(n_samples, 1);
    
    for i = 1:n_samples
        x = X_new(:, i);
        Y1_hat = pinv(D1' * D1) * D1' * x;
        Y2_hat = pinv(D2' * D2) * D2' * x;
        X_new_hat = a * D1 * Y1_hat + b * D2 * Y2_hat;
        T2_stats(i) = a * (Y1_hat' * Y1_hat) + b * (Y2_hat' * Y2_hat);
        SPE_stats(i) = (x - X_new_hat)' * (x - X_new_hat);
    end
end

function RelErr = calculate_relerr(W1_k, W1_k1, Y1_k, Y1_k1, W2_k, W2_k1, Y2_k, Y2_k1)
    relerr_W1 = norm(W1_k1 - W1_k, 'fro') / (norm(W1_k, 'fro') + 1);
    relerr_Y1 = norm(Y1_k1 - Y1_k, 'fro') / (norm(Y1_k, 'fro') + 1);
    relerr_W2 = norm(W2_k1 - W2_k, 'fro') / (norm(W2_k, 'fro') + 1);
    relerr_Y2 = norm(Y2_k1 - Y2_k, 'fro') / (norm(Y2_k, 'fro') + 1);
    RelErr = max([relerr_W1, relerr_Y1, relerr_W2, relerr_Y2]);
end

