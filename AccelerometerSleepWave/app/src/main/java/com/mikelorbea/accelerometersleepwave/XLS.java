package com.mikelorbea.accelerometersleepwave;

import android.content.Context;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFCellStyle;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hssf.util.HSSFColor;
import org.apache.poi.ss.usermodel.CellStyle;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;

public class XLS {

    public boolean exportDataAsXLS(final ArrayList<Long> timestamp, final ArrayList<Float> x_axis,
                                   final ArrayList<Float> y_axis, final ArrayList<Float> z_axis,
                                   String title, final String route, final String[] columnTitles, final Context context) {

        //Creates the folder
        File folder = new File(route);

        if (!folder.exists()) {
            if(!folder.mkdir()) {
                return false;
            }
        }

        //Creates the xls file
        HSSFWorkbook hwb = new HSSFWorkbook();
        HSSFSheet sheet = hwb.createSheet(title);

        //Creates the style for the column titles
        CellStyle style = hwb.createCellStyle();
        style.setFillForegroundColor(HSSFColor.LIGHT_BLUE.index);
        style.setFillPattern(HSSFCellStyle.SOLID_FOREGROUND);

        //For each row of the database
        for (int x = 0; x < 4; x++) {

            //Creates the row
            HSSFRow row = sheet.createRow(x);

            //For each cell
            for (int i = 0; i < timestamp.size() + 1; i++) {

                //Creates the cell and introduce the data
                HSSFCell cell = row.createCell(i);
                String data;

                if(i == 0) {
                    switch (x) {
                        case 0:
                            data = columnTitles[0];
                            break;

                        case 1:
                            data = columnTitles[1];
                            break;

                        case 2:
                            data = columnTitles[2];
                            break;

                        default:
                            data = columnTitles[3];
                            break;
                    }
                    cell.setCellStyle(style);
                }
                else {
                    switch (x) {
                        case 0:
                            data = timestamp.get(i-1).toString();
                            break;

                        case 1:
                            data = x_axis.get(i-1).toString();
                            break;

                        case 2:
                            data = y_axis.get(i-1).toString();
                            break;

                        default:
                            data = z_axis.get(i-1).toString();
                            break;
                    }
                }

                cell.setCellValue(data);
            }
        }

        //Save the file
        try {
            FileOutputStream fileOut = new FileOutputStream(route + "/" + "movement_data_gathering.xls");
            hwb.write(fileOut);
            fileOut.close();
            return true;
        }
        //If some error occurs
        catch (Exception e) {
            return false;

        }
    }
}
