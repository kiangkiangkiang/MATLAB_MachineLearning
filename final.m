clear;
load dataset6
gscatter(D.train.x(:,1),D.train.x(:,2),D.train.labels,'br','ox');
repeats=100;k_near1=10;k_near2=20;
trainErr.LDA = zeros(repeats,1);
trainErr.QDA = zeros(repeats,1);
trainErr.knn = zeros(repeats,1);
testErr.LDA = zeros(repeats,1);
testErr.QDA = zeros(repeats,1);
testErr.knn = zeros(repeats,1);
N_train=200;N_test=200;
Ntr=size(D.train.x,1);
Nte=size(D.test.x,1);
trainFcn = 'trainscg';
hiddenLayerSize=10;
net = patternnet(hiddenLayerSize,trainFcn);
net.divideParam.trainRatio=70/100;
net.divideParam.valRatio=15/100;
net.divideParam.testRatio=15/100;
%--------
tic
for i = 1:repeats
    indexTr = randperm(Ntr);    
    Xtrain = D.train.x(indexTr(1:N_train),:);
    Ytrain = D.train.labels(indexTr(1:N_train),:);    
    indexTe = randperm(Nte);
    Xtest = D.test.x(indexTe(1:N_test),:);
    Ytest = D.test.labels(indexTe(1:N_test),:);
    
    temp = Ytrain +1;
    YANNtrain = full(ind2vec(temp'))';
    temp = Ytest +1 ;
    YANNtest = full(ind2vec(temp'))';
    
    %LDA
    LDA = fitcdiscr(Xtrain,Ytrain);
    trainErr.LDA(i) = resubLoss(LDA);
    M = predict(LDA,Xtest);
    testErr.LDA(i) = 1-sum(M==Ytest)/N_test;
    
    %QDA
    QDA = fitcdiscr(Xtrain,Ytrain,'DiscrimType','quadratic');
    trainErr.QDA(i) = resubLoss(QDA);
    M = predict(QDA,Xtest);
    testErr.QDA(i) = 1-sum(M==Ytest)/N_test;
    
    %knn10
    knn = fitcknn(Xtrain,Ytrain,"NumNeighbors",k_near1);%k=5
    trainErr.knn1(i) = resubLoss(knn);
    M = predict(knn,Xtest);
    testErr.knn1(i) = 1-sum(M==Ytest)/N_test;
    
    %knn20
    knn = fitcknn(Xtrain,Ytrain,"NumNeighbors",k_near2);%k=5
    trainErr.knn2(i) = resubLoss(knn);
    M = predict(knn,Xtest);
    testErr.knn2(i) = 1-sum(M==Ytest)/N_test;
    
    %ann
    x = Xtrain';
    t = YANNtrain';
    [net,tr] = train(net,x,t);
    y = net(x);
    trainErr.ann(i)=tr.best_perf;
    testErr.ann(i)=tr.best_tperf;
        
    %reg
    reg=fitlm(Xtrain,Ytrain);
    yHat = reg.Fitted;
    yHat(yHat<0.5)=0;yHat(yHat>0.5)=1;
    trainErr.reg(i) = sum(abs(Ytrain-yHat))/N_train;
    M = predict(reg,Xtest);
    M(M<0.5)=0;M(M>0.5)=1;
    testErr.reg(i) = 1-sum(M==Ytest)/N_test;
    
    %reg*
    reg=fitlm(Xtrain,Ytrain,'quadratic');
    yHat = reg.Fitted;
    yHat(yHat<0.5)=0;yHat(yHat>0.5)=1;
    trainErr.Areg(i) = sum(abs(Ytrain-yHat))/N_train;
    M = predict(reg,Xtest);
    M(M<0.5)=0;M(M>0.5)=1;
    testErr.Areg(i) = 1-sum(M==Ytest)/N_test;
    
end
toc
fprintf("train_reg : %.2f\n",100*(1-mean(trainErr.reg)));
fprintf("train_Areg : %.2f\n",100*(1-mean(trainErr.Areg)));
fprintf("train_LDA : %.2f\n",(1-mean(trainErr.LDA))*100);
fprintf("train_QDA : %.2f\n",(1-mean(trainErr.QDA))*100);
fprintf("train_knn%d: %.2f\n",k_near1,100*(1-mean(trainErr.knn1)));
fprintf("train_knn%d: %.2f\n",k_near2,100*(1-mean(trainErr.knn2)));
fprintf("train_ann : %.2f\n\n",100*(1-mean(trainErr.ann)));

fprintf("test_reg : %.2f\n",100*(1-mean(testErr.reg)));
fprintf("test_Areg : %.2f\n",100*(1-mean(testErr.Areg)));
fprintf("test_LDA : %.2f\n",100*(1-mean(testErr.LDA)));
fprintf("test_QDA : %.2f\n",100*(1-mean(testErr.QDA)));
fprintf("test_knn%d: %.2f\n",k_near1,100*(1-mean(testErr.knn1)));
fprintf("test_knn%d : %.2f\n",k_near2,100*(1-mean(testErr.knn2)));
fprintf("test_ann : %.2f\n",100*(1-mean(testErr.ann)));













