import java.io.FileReader;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;

public class ParseData {
    
    public static void main(String[] args) {
         String file1 = args[0];
         String file2 = args[1];
        
         HashSet<Integer> top25 = new HashSet<Integer>();
         top25.add(183);
         top25.add(82);
         top25.add(603);
         //retrive top 25 symbols
         /*
         try {
            FileReader fr = new FileReader(file1);
            BufferedReader br = new BufferedReader(fr);

            String store;
            while(true) {
                try {
                    store = br.readLine();
                    if (store == null) break;
                    top25.add(Integer.parseInt(store));
                }
                catch (IOException io) {
                    System.out.println("IOException");
                }
             }

             //for (int item : top25) {
             //   System.out.println(item + "");
             //}
         }
         catch (FileNotFoundException ex)  {
            System.out.println("FileNotFoundException");
         }
         */
         

        //extract and process train data for top 25 symbols
        
        try {
            FileReader fr = new FileReader(file2);
            BufferedReader br = new BufferedReader(fr);

            String line;
            try {
                line = br.readLine(); //remove header
            }
            catch (IOException io) {
                System.out.println("IOException");
            }

            try {
                while((line = br.readLine()) != null) {
                
                    String id = "";
                    char c;
                    int i = 0;

                    while((c = line.charAt(i)) != ';') {
                        id += c; 
                        i++; 
                    }

                    i = Integer.parseInt(id);
                    if (!top25.contains(i)) continue;


                    //row for a top25 symbol has been found
                    String[] fields = line.split(";");

                    try{
                        PrintWriter writer = new PrintWriter(fields[0] + ".text", "UTF-8");
                        
                        writer.println(fields[0]); //id of symbol
                        writer.println(fields[2]); //2D Array of Dictionaries

                        while(true) { //keep reading till symbol's block ends
                            int bsd;
                            id = "";
                            while ((bsd = br.read()) != ';') {
                                if (bsd == -1) break;
                                id += (char) bsd;
                            }
                            i = Integer.parseInt(id);
                            if (!top25.contains(i)) break;
                            else {
                                fields = br.readLine().split(";");
                                writer.println(fields[1]); //print next 2D array
                            }
                        }

                        writer.close();
                    } catch (IOException e) {
                        System.out.println("IOException");
                    }
                }   
            }
            catch (IOException io) {
                System.out.println("IOException");
            }
        }
        catch (FileNotFoundException e) {
            System.out.println("FileNotFoundException Hi");
        }
    }
}


