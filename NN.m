files = dir('mendata1s/*.wav');
unstutteredFeatures = [];
stutteredFeatures = [];
targetVector = [];

for i = 1:670
    fileName = strcat('mendata1s/',files(i).name);
    if fileName(length(fileName)-4) == 'U' %unstuttered speech is 0, stuttered speech is 1
       [y1,~] = audioread(fileName);
       unstutteredFeatures = vertcat(mean(y1),unstutteredFeatures);
       targetVector = [targetVector ; 0];
    else
       [y2,~] = audioread(fileName); 
       stutteredFeatures = vertcat(mean(y2),stutteredFeatures);
       targetVector = [targetVector ; 1];
    end
end

inputMatrix = vertcat(unstutteredFeatures,stutteredFeatures);
 
net = patternnet(6);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:370;
net.divideParam.valInd = 370:570;
net.divideParam.testInd = 571:670;

[net,tr] = train(net, inputMatrix', targetVector');
outputs = net(inputMatrix');
per = perform(net, targetVector', outputs);

trainTargets = targetVector' .* tr.trainMask{1};
valTargets   = targetVector' .* tr.valMask{1};
testTargets  = targetVector' .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,outputs);
valPerformance   = perform(net,valTargets,outputs);
testPerformance  = perform(net,testTargets,outputs);

