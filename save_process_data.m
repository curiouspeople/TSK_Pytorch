datafileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\Raw_Data');
dirOutput=dir(fullfile(datafileFolder,'*.mat'));
dataFileNames={dirOutput.name};
LabelfileFolder=fullfile('D:\PyProject\深度学习\SEED-VIG\perclos_labels');
LabelOutput=dir(fullfile(LabelfileFolder,'*.mat'));
LabelFileNames={LabelOutput.name};
CrossSub=zeros(23,1);
epsilon=1e-8;
for i=1:23
    disp(i)

%     %对原始数据的处理
%     cro_data=load(strcat('D:\PyProject\深度学习\SEED-VIG\Raw_Data\',dataFileNames{i})).EEG.data;
%     temp=reshape(cro_data,885,27200);
%     [coeff,score,latent,tsquared,explained,mu] = pca(temp);
%     for j=1:length(explained)
%         if sum(explained(1:j))>99
%             an=j;
%             break;
%         end
%     end
%     cro_X=score(:,1:40);
%     cro_X = zscore(cro_X);
%     %% fcm
%     [C0,U,Obj]=fcm(cro_X,40);
%     cro_X=(C0*U)';
%     cro_X = zscore(cro_X);
% % 
%     cro_X = zscore(cro_X); 
%     save(['D:\PyProject\深度学习\SEED-VIG\self_process\',num2str(i),'.mat'],'cro_X')
    y_label=load(strcat('D:\PyProject\深度学习\SEED-VIG\perclos_labels\', LabelFileNames{i})).perclos;
    for k=1:885
        if y_label(k)>0.35
            y_label(k)=2;
        else
            y_label(k)=1;
        end
    end
    save(['D:\PyProject\深度学习\SEED-VIG\self_processLabel\',num2str(i),],'y_label')
end