function [Model] = train(X,Y)
    % X feature
    % Y label
    [time, acc, Model] = elm_train(Y, X, 1, 6400, 'tribas');
    save('Model.mat', 'Model');
end


