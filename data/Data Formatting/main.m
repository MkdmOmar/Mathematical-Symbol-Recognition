numLabels = 24;
C = cell(numLabels, 1);
A = {81,82,88,116,166,176,182,183,191,193,513,524,536,601,603,618,698,753,882,888,891,944,968,1066};

for i = 1:numLabels
    C{i, 1} = csvToMatrix(strcat(strcat('matricies/data3/',int2str(A{i})),'mat.txt'));
end    


fidTrain=fopen('train.txt','wt');
fidLabel=fopen('trainLabel.txt','wt');
fidTest=fopen('test.txt','wt');
fidTestLabel=fopen('testLabel.txt','wt');

for j = 1:numLabels
    data = C{j, 1};
    testIndicies = randperm(size(data,1),int64(size(data,1)/5));
    % -2 because we have 2 extra points at the end
    for i = 1:(size(data,1) - 2)
        I = mat2gray(data{i,1});
        if (and((size(I,1) > 34),(size(I,2) > 34)))
            mat = matCleaner(I);
            IOut = convertToNxN(I, 35);
            IFlat = reshape(IOut.',1,[]);

            %See if any in test index
            if (any(abs(testIndicies-i)<1e-10))
                fprintf(fidTest,'%d ',IFlat);
                fprintf(fidTest,'\n');
                fprintf(fidTestLabel,'%d\n',j);
            else 
                fprintf(fidTrain,'%d ',IFlat);
                fprintf(fidTrain,'\n');
                fprintf(fidLabel,'%d\n',j);

            end
               
        end
            
    end
end

fclose(fidTrain);
fclose(fidLabel);
fclose(fidTest);
fclose(fidTestLabel);
