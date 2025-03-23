% 读取原始 CSV 文件
filePath = 'all_results.csv'; % 替换为你的文件路径
resultsTable = readtable(filePath); % 读取 CSV 文件为 table

% 显示原始表格
disp('原始表格：');
disp(resultsTable);

% 需要转换为百分数的列名
columnsToConvert = {'T2_FDR', 'SPE_FDR', 'T2_FAR', 'SPE_FAR'};

% 将指定列的小数转换为带 % 的字符串
for col = columnsToConvert
    resultsTable.(col{1}) = arrayfun(@(x) sprintf('%.2f%%', x * 100), ...
        resultsTable.(col{1}), 'UniformOutput', false);
end

% 显示转换后的表格
disp('转换为带 % 的表格：');
disp(resultsTable);

% 保存到新的 CSV 文件
outputFilePath = 'Documents/MATLAB/LRSDFS/all_results_percent_with_symbol.csv';
writetable(resultsTable, outputFilePath);
disp(['结果已保存到: ' outputFilePath]);