function labels = classify2(Model, X)

tfeas = [];
for i = 1 : size(X,1)
    tfeas = [tfeas;naivehog(reshape(X(i,:),[32,32,3]))'];
end

TV.P=tfeas';

NumberofTestingData=size(TV.P,2);

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=double(Model.InputWeight)*TV.P;
clear TV.P;             %   Release input of testing data
ind=ones(1,NumberofTestingData);
BiasMatrix=Model.BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(Model.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here

end
TY=(H_test' * Model.OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

[~, labels]= max(TY);
labels = labels' -1 ;

end

% fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

