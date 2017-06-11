files = dir('data1s/*.wav');
unstutteredFeatures = [];
stutteredFeatures = [];
targetVector = [];

for i = 1:274
    fileName = strcat('data1s/',files(i).name);
    if fileName(length(fileName)-4) == 'U' %unstuttered speech is 0, stuttered speech is 1
       [y1,~] = audioread(fileName);
       len = length(y1);
       %divide y1 into 8 vectors of size len/8
       y01 = y1(1:len/8);
       y02 = y1((len/8)+1:len/4);
       y03 = y1((len/4)+1:3*len/8);
       y04 = y1((3*len/8)+1:len/2);
       y05 = y1(len/2+1:5*len/8);
       y06 = y1(5*len/8+1:6*len/8);
       y07 = y1(6*len/8+1:7*len/8);
       y08 = y1(7*len/8+1:len);
       unstutteredFeatures = vertcat(mean(y01),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y02),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y03),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y04),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y05),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y06),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y07),unstutteredFeatures);
       unstutteredFeatures = vertcat(mean(y08),unstutteredFeatures);
       
       for j = 1:8
          targetVector = [targetVector ; 0];
       end 

    else
       [y2,~] = audioread(fileName);
       len = length(y2);
       %divide y2 into 8 vectors of size len/8
       y01 = y2(1:len/8);
       y02 = y2((len/8)+1:len/4);
       y03 = y2((len/4)+1:3*len/8);
       y04 = y2((3*len/8)+1:len/2);
       y05 = y2(len/2+1:5*len/8);
       y06 = y2(5*len/8+1:6*len/8);
       y07 = y2(6*len/8+1:7*len/8);
       y08 = y2(7*len/8+1:len);
       stutteredFeatures = vertcat(mean(y01),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y02),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y03),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y04),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y05),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y06),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y07),stutteredFeatures);
       stutteredFeatures = vertcat(mean(y08),stutteredFeatures);
       
       for j = 1:8
          targetVector = [targetVector ; 1];
       end 

    end
end

inputMatrix = vertcat(unstutteredFeatures,stutteredFeatures);
 
net = patternnet(4);
net = train(net, inputMatrix', targetVector');
y3 = net(inputMatrix');
per = perform(net, targetVector', y3);
