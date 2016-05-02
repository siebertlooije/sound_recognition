package com.mikelorbea.accelerometersleepwave;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;

import java.util.ArrayList;

/**
 * Created by miors on 28/04/2016.
 */
public class ChartActivity extends Activity{
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chart);

        LineChart lineChart = (LineChart) findViewById(R.id.chart);
        ArrayList<Float> x_axis = (ArrayList<Float>) getIntent().getSerializableExtra("x_data");
        ArrayList<Float> y_axis = (ArrayList<Float>) getIntent().getSerializableExtra("y_data");
        ArrayList<Float> z_axis = (ArrayList<Float>) getIntent().getSerializableExtra("z_data");
        ArrayList<Long> timestamps = (ArrayList<Long>) getIntent().getSerializableExtra("timestamps");

        ArrayList<Entry> x_entries = new ArrayList<>();
        int count = 0;
        for(float x : x_axis) {
            x_entries.add(new Entry(x, count));
            ++count;
        }
        LineDataSet x_dataset = new LineDataSet(x_entries, "x-axis");

        ArrayList<Entry> y_entries = new ArrayList<>();
        count = 0;
        for(float y : y_axis) {
            y_entries.add(new Entry(y, count));
            ++count;
        }
        LineDataSet y_dataset = new LineDataSet(y_entries, "y-axis");


        ArrayList<Entry> z_entries = new ArrayList<>();
        count = 0;
        for(float z : z_axis) {
            z_entries.add(new Entry(z, count));
            ++count;
        }
        LineDataSet z_dataset = new LineDataSet(z_entries, "z-axis");


        ArrayList<String> labels = new ArrayList<>();
        for(long time : timestamps) {
            labels.add(String.valueOf(time));
        }

        x_dataset.setDrawCubic(true);
        x_dataset.setColor(Color.RED);

        y_dataset.setDrawCubic(true);
        y_dataset.setColor(Color.GREEN);

        z_dataset.setDrawCubic(true);
        z_dataset.setColor(Color.BLUE);

        ArrayList<ILineDataSet> axisDataSets = new ArrayList<>();
        axisDataSets.add(x_dataset);
        axisDataSets.add(y_dataset);
        axisDataSets.add(z_dataset);

        LineData data = new LineData(labels, axisDataSets);
        lineChart.setData(data);
        lineChart.setDescription("Movement");
    }
}
