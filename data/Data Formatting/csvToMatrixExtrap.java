import java.util.Arrays;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.JFrame;


public class csvToMatrixExtrap{
	public static void main(String[] args){
		
		int labelID = StdIn.readInt();
		String all = StdIn.readAll();
		String samples[] = all.split(";");	
		boolean first = true;
		int x, y;
		int prevX, prevY;
		int difX, difY;
		double slope;
		int maxx, maxy, minx, miny;
		int length = 3000;
		int width = 3000;

		//System.out.println(samples[0]);
		//System.out.println(labelID);
		for(int i = 0; i < samples.length; i++){
			String[] strokes = samples[i].split("-");
			maxx = 0;
			minx = length;
			maxy = 0;
			miny = width;
			int matrix[][] = new int[length][width];
			for(int l = 0; l < strokes.length; l++){	
				first = true;
				String[] lines = strokes[l].split("\n");
				

				prevX = prevY = x = y = 0;
				for(int j = 0; j < lines.length; j++){
					String nums[] = lines[j].split(",");
					if(nums.length > 1){
						
						if (!first){
							prevX = x;
							prevY = y;
						}
						x = (int)Double.parseDouble(nums[0]);
						y = (int)Double.parseDouble(nums[1]);

						matrix[y][x] = 1;

						if (!first){
							difX = x - prevX;
							difY = y - prevY;
							slope = 1.0*difY/difX;
							if (difX < 0){
								for (int start = difX; start < 0; start++){
									matrix[(int)Math.round(slope*start + prevY)][prevX+start] = 1;
								}
							}
							else{
								for (int start = difX; start > 0; start--){
									matrix[(int)Math.round(slope*start + prevY)][prevX+start] = 1;
								}
							}

							slope = 1.0*difX/difY;

							if (difY < 0){
								for (int start = difY; start < 0; start++){
									matrix[prevY+start][(int)Math.round(slope*start + prevX)] = 1;
								}
							}
							else{
								for (int start = difY; start > 0; start--){
									matrix[prevY+start][(int)Math.round(slope*start + prevX)] = 1;
								}
							}
							
						}
						if (x > maxx)
							maxx = x;
						if (x < minx)
							minx = x;
						if (y > maxy)
							maxy = y;
						if (y < miny)
							miny = y;

						first = false;
					}

				}
			}

			for (int m = miny; m < maxy + 1; m++){
				for (int n = minx; n < maxx + 1; n++){
					System.out.print(matrix[m][n] + ",");
					
				}
				System.out.println();
								
			} 
			System.out.println();


			//System.out.println(Arrays.deepToString(matrix));
			//System.out.println(i);
		


		}


	}
}