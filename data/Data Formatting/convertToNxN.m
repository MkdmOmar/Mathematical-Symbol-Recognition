function Iout = convertToNxN(I, N)
    
[length,width] = size(I);

lScale = (N-1)/(length - 1);
wScale = (N-1)/(width - 1);

Iout = zeros(N);
for i = 1:length
   for j = 1:width 
      if I(i,j) == 1;
          Iout(round((i-1)*lScale + 1),round((j-1)*wScale + 1)) = 1;
          
      end   
       
   end
    
end