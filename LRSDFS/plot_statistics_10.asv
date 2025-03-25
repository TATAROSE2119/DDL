ICA_IDV_10_SPE_statistics = load('LRSDFS/IDV4_IDV10/ICA/IDV10_SPE_statistics.mat');
ICA_IDV_10_SPE_limits = load('LRSDFS/IDV4_IDV10/ICA/IDV10_SPE_limit.mat');
ICA_IDV_10_T2_statistics = load('LRSDFS/IDV4_IDV10/ICA/IDV10_T2_statistics.mat');
ICA_IDV_10_T2_limits = load('LRSDFS/IDV4_IDV10/ICA/IDV10_T2_limit.mat');

PCA_IDV_10_SPE_statistics = load('LRSDFS/IDV4_IDV10/PCA/IDV10_SPE_statistics.mat');
PCA_IDV_10_SPE_limits = load('LRSDFS/IDV4_IDV10/PCA/IDV10_SPE_limit.mat');
PCA_IDV_10_T2_statistics = load('LRSDFS/IDV4_IDV10/PCA/IDV10_T2_statistics.mat');
PCA_IDV_10_T2_limits = load('LRSDFS/IDV4_IDV10/PCA/IDV10_T2_limit.mat');

LDL_IDV_10_SPE_statistics = load('LRSDFS/IDV4_IDV10/LDL/IDV10_SPE_statistics.mat');
LDL_IDV_10_SPE_limits = load('LRSDFS/IDV4_IDV10/LDL/IDV10_SPE_limit.mat');
LDL_IDV_10_T2_statistics = load('LRSDFS/IDV4_IDV10/LDL/IDV10_T2_statistics.mat');
LDL_IDV_10_T2_limits = load('LRSDFS/IDV4_IDV10/LDL/IDV10_T2_limit.mat');

SDL_IDV_10_SPE_statistics = load('LRSDFS/IDV4_IDV10/SDL/IDV10_SPE_statistics.mat');
SDL_IDV_10_SPE_limits = load('LRSDFS/IDV4_IDV10/SDL/IDV10_SPE_limit.mat');
SDL_IDV_10_T2_statistics = load('LRSDFS/IDV4_IDV10/SDL/IDV10_T2_statistics.mat');
SDL_IDV_10_T2_limits = load('LRSDFS/IDV4_IDV10/SDL/IDV10_T2_limit.mat');

LSDDL_IDV_10_SPE_statistics = load('LRSDFS/IDV4_IDV10/LSDDL/IDV10_SPE_statistics.mat');
LSDDL_IDV_10_SPE_limits = load('LRSDFS/IDV4_IDV10/LSDDL/IDV10_SPE_limit.mat');
LSDDL_IDV_10_T2_statistics = load('LRSDFS/IDV4_IDV10/LSDDL/IDV10_T2_statistics.mat');
LSDDL_IDV_10_T2_limits = load('LRSDFS/IDV4_IDV10/LSDDL/IDV10_T2_limit.mat');

%% Plotting 将两种方法的统计图画在一起
%% IDV 10--------------------------------------------------------------
%-----------------------------ICA-------------------------------------
figure;
subplot(2,1,1);
plot(ICA_IDV_10_T2_statistics.T2_ICA, 'b', 'LineWidth', 2);
hold on; 
yline(ICA_IDV_10_T2_limits.T2_ICA_limit, 'r--', 'LineWidth', 2);
ylabel('T^2');

subplot(2,1,2);
plot(ICA_IDV_10_SPE_statistics.SPE_ICA, 'b', 'LineWidth', 2);
hold on;
yline(ICA_IDV_10_SPE_limits.SPE_ICA_limit, 'r--', 'LineWidth', 2);
ylabel('SPE');

sgtitle('(b)','FontName', 'Times New Roman', 'FontSize', 14);
saveas(gcf, 'IDV4_IDV10/plot/10_ICA.svg');
%-----------------------------ICA-------------------------------------
%-----------------------------PCA-------------------------------------
figure;
subplot(2,1,1);
plot(PCA_IDV_10_T2_statistics.T2_PCA, 'b', 'LineWidth', 2);
hold on;
yline(PCA_IDV_10_T2_limits.T2_PCA_limit, 'r--', 'LineWidth', 2);
ylabel('T^2');

subplot(2,1,2);
plot(PCA_IDV_10_SPE_statistics.SPE_PCA, 'b', 'LineWidth', 2);
hold on;
yline(PCA_IDV_10_SPE_limits.SPE_PCA_limit, 'r--', 'LineWidth', 2);
ylabel('SPE');

sgtitle('(a)','FontName', 'Times New Roman', 'FontSize', 14);
saveas(gcf, 'IDV4_IDV10/plot/10_PCA.svg');
%-----------------------------PCA-------------------------------------
%-----------------------------LDL-------------------------------------
figure;
subplot(2,1,1);
plot(LDL_IDV_10_T2_statistics.T2_statistics, 'b', 'LineWidth', 2);
hold on;
yline(LDL_IDV_10_T2_limits.T2_limit, 'r--', 'LineWidth', 2);
ylabel('T^2');

subplot(2,1,2);
plot(LDL_IDV_10_SPE_statistics.SPE_statistics, 'b', 'LineWidth', 2);
hold on;
yline(LDL_IDV_10_SPE_limits.SPE_limitCopy, 'r--', 'LineWidth', 2);
ylabel('SPE');

sgtitle('(c)','FontName', 'Times New Roman', 'FontSize', 14);
saveas(gcf, 'IDV4_IDV10/plot/10_LDL.svg');
%-----------------------------LDL-------------------------------------
%-----------------------------SDL-------------------------------------
figure;
subplot(2,1,1);
plot(SDL_IDV_10_T2_statistics.T2_statistics, 'b', 'LineWidth', 2);
hold on;
yline(SDL_IDV_10_T2_limits.T2_limit, 'r--', 'LineWidth', 2);
ylabel('T^2');

subplot(2,1,2);
plot(SDL_IDV_10_SPE_statistics.SPE_statistics, 'b', 'LineWidth', 2);
hold on;
yline(SDL_IDV_10_SPE_limits.SPE_limit, 'r--', 'LineWidth', 2);
ylabel('SPE');

sgtitle('(d)','FontName', 'Times New Roman', 'FontSize', 14);
saveas(gcf, 'IDV4_IDV10/plot/10_SDL.svg');
%-----------------------------SDL-------------------------------------
%-----------------------------LSDDL-------------------------------------
figure;
subplot(2,1,1);
plot(LSDDL_IDV_10_T2_statistics.T2_statistics, 'b', 'LineWidth', 2);
hold on;
yline(LSDDL_IDV_10_T2_limits.T2_limit, 'r--', 'LineWidth', 2);
ylabel('T^2');

subplot(2,1,2);
plot(LSDDL_IDV_10_SPE_statistics.SPE_statistics, 'b', 'LineWidth', 2);
hold on;   
yline(LSDDL_IDV_10_SPE_limits.SPE_limit, 'r--', 'LineWidth', 2);
ylabel('SPE');

sgtitle('(e)','FontName', 'Times New Roman', 'FontSize', 14);
saveas(gcf, 'IDV4_IDV10/plot/10_LSDDL.svg');
%-----------------------------LSDDL-------------------------------------





   