clear all;close all;

confusion_mat=zeros(10,10);
myDir='Final_digits/';
myFiles = dir(fullfile(myDir,'*.mat'));
for i = 1:length(myFiles)
    Tname = myFiles(i).name;
    T_actual=str2num(Tname(1));
    Tname=strcat('Final_digits/',Tname);
    T=load(Tname);
    T=cell2mat(struct2cell(T));
    T=squeeze(T);
    utterances=zeros(299,2);
    uter=0;
    for j=1:length(myFiles)
        uter=uter+1;
        if(i==j)
            uter=uter-1;
            continue;
        end
        Rname = myFiles(j).name;
        R_actual=str2num(Rname(1));
        Rname=strcat('Final_digits/',Rname);
        R=load(Rname);
        R=cell2mat(struct2cell(R));
        R=squeeze(R);
        
        %actual code here
        M=size(T,1);
        N=size(R,1);
        %STEP 1: generate (dis)similiarity matrix using euclidean dist
        S=zeros(M,N);
        for m=1:M
            for n=1:N
                S(m,n)=norm(T(m,:)-R(n,:));
            end
        end
        %Step 2: generate DTW matrix (D)
        D=zeros(M,N);
        tracking=zeros(M,N);
        for m=1:M
            for n=1:N
                if m==1 && n==1
                    tracking(m,n)=0;
                    D(m,n)=S(m,n);
                elseif m==1 && n~=1
                    tracking(m,n)=2;
                    D(m,n)=S(m,n)+D(m,n-1);
                elseif m~=1 &&n==1
                    tracking(m,n)=1;
                    D(m,n)=S(m,n)+D(m-1,n);
                else
                    [val,tracking(m,n)]=min([D(m-1,n),D(m,n-1),D(m-1,n-1)]);
                    D(m,n)=S(m,n)+val;
                end
            end
        end
        %STEP 3: Backtrack to find optimal path
        M_cur=M;
        N_cur=N;
        disim_sum=[];
        path_length=[];
        it=0;
        while true
            if tracking(M_cur,N_cur)==1
                M_cur=M_cur-1;
            elseif tracking(M_cur,N_cur)==2
                N_cur=N_cur-1;
            elseif tracking(M_cur,N_cur)==3
                M_cur=M_cur-1;
                N_cur=N_cur-1;
            elseif tracking(M_cur,N_cur)==0
                break;
            end
            it=it+1;
            disim_sum(it)=S(M_cur,N_cur);
            path_length(it)=D(M_cur,N_cur);
        end
        disim_sum=flip(disim_sum);
        path_length=flip(path_length);
        for ind=1:it-1
            path_length(ind+1:end)=path_length(ind+1:end)-path_length(ind);
        end
        path_sum=sum(path_length);
        %normalize disimularity matrix with path length
        for ind=1:it
            disim_sum(ind)=disim_sum(ind)*path_length(ind)/path_sum;
        end
        %add utterance expected and actual
        utterances(uter,1)=sum(disim_sum);
        utterances(uter,2)=R_actual;
    end
    %generate confusion matrix
    utterances=sortrows(utterances);
    utterances=utterances(1:29,:);
    for ut=1:29
        confusion_mat(utterances(ut,2)+1,T_actual+1)=confusion_mat(utterances(ut,2)+1,T_actual+1)+1;
    end
end

col_names={'input_0','input_1','input_2','input_3','input_4','input_5','input_6','input_7','input_8','input_9'};
row_names={'output_0','output_1','output_2','output_3','output_4','output_5','output_6','output_7','output_8','output_9'};
conf_table=array2table(confusion_mat,'RowNames',row_names,'VariableNames',col_names)
