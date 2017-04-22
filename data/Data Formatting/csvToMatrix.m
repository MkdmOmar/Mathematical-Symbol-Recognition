function data = csvToMatrix(filename)


%filename = 'matricies/data3/1066mat.txt';

%# Read all the data
allData = csvread(filename);

%# Compute the empty line indices
fid = fopen(filename);
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
blankLines = find(cellfun('isempty', lines{1}))';

%# Find the indices to separate data into cells from the whole matrix
iEnd = [blankLines - (1:length(blankLines)) size(allData,1)];
iStart = [1 (blankLines - (0:length(blankLines)-1))];
nRows = iEnd - iStart + 1;

%# Put the data into cells
data = mat2cell(allData, nRows)