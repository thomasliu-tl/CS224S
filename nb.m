nb2 = fitNaiveBayes(inputMatrix, targetVector);
p = nb2.predict(inputMatrix);
cMat = confusionmat(targetVector, p);
 
cMat
