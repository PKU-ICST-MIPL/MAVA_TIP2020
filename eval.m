% clear;

te_n_I = 1000;
te_n_T = 5000;
I_T = 5;

global_mat = 'global/flickr_global.mat';
disp(['loading global mat.....',global_mat]);
load(global_mat);
disp('loading global mat completed.....');
s_global=test_txt*test_img';

local_mat = 'local/flickr_local.mat';
disp(['loading local mat.....',local_mat]);
load(local_mat);
disp('loading local mat completed.....');
s_local = similarity;

relation_mat = 'relation/flickr_relation.mat';
disp(['loading relation mat.....',relation_mat]);
load(relation_mat);
disp('loading relation mat completed.....');
s_relation = similarity;

W=25*s_local+5*s_relation+1*s_global;

x=[1,5,10];
[Y,ImgQuery] = sort(W',2,'descend');
for i=1:3
    R=x(i);
    res = ImgQuery(:,1:R);
    res_i_t = res;
    cnt = 0;
    for ii=1:te_n_I
        for j=1:R
            if I_T ==1
                if res(ii,j)==ii
                    cnt = cnt+1;
                    break;
                end
            else
                if (ii*5)>=res(ii,j)&&((ii-1)*5)<res(ii,j)
                    cnt = cnt+1;
                    break;
                end
            end
        end
    end
    disp(['R@' num2str(R) ':' num2str(cnt/te_n_I)]);
end

[Y,ImgQuery] = sort(W,2,'descend');
for i=1:3
    R=x(i);
    res = ImgQuery(:,1:R);
    res_t_i = res;
    cnt = 0;
    for ii=1:te_n_T
        for j=1:R
            if I_T ==1
                if res(ii,j)==ii
                    cnt = cnt+1;
                    break;
                end
            else
                if (res(ii,j)*5)>=ii&&((res(ii,j)-1)*5)<ii
                    cnt = cnt+1;
                    break;
                end
            end
        end
    end
    disp(['R@' num2str(R) ':' num2str(cnt/te_n_T)]);
end