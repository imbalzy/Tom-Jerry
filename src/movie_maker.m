clear all;
close all;

folder = '02_Dec_2021_03_14_05';
epis_num = '00001';

path = ['../results/' folder  '/figures'];
save_path = ['../results/' folder  '/video/'];

mkdir(save_path)

v = VideoWriter([save_path, 'epis', epis_num,'_anim'],'MPEG-4');
v.FrameRate = 10;
open(v)

S = dir(fullfile(path,['epis',epis_num,'_*.png'])); % pattern to match filenames.
for k = 1:numel(S)
    F = fullfile(path,S(k).name);
    I = imread(F);
    writeVideo(v,I)
end

close(v)