
SVMModel = fitcsvm(inputMatrix, targetVector);
CVSVMModel = crossval(SVMModel);
missClass = kfoldLoss(CVSVMModel);

missClass

