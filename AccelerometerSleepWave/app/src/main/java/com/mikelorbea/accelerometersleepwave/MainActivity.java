package com.mikelorbea.accelerometersleepwave;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AlertDialog;
import android.view.View;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;

import jp.co.recruit_lifestyle.android.widget.PlayPauseButton;

public class MainActivity extends Activity implements SensorEventListener{

    private SensorManager senSensorManager;
    private Sensor senAccelerometer;
    private long lastUpdate;
    private long firstTime;
    private ArrayList<Long> timestamps;
    private ArrayList<Float> x_axis, y_axis, z_axis;

    private PlayPauseButton playPauseButton;
    private boolean isRecording = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        isRecording = false;
        senSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        senAccelerometer = senSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        playPauseButton = (PlayPauseButton) findViewById(R.id.main_play_pause_button);
        playPauseButton.setColor(Color.YELLOW);
        playPauseButton.setOnControlStatusChangeListener(new PlayPauseButton.OnControlStatusChangeListener() {
            @Override
            public void onStatusChange(final View view, boolean state) {
                if (state) {
                    playPauseButton.setColor(Color.RED);
                    initialitze();
                    senSensorManager.registerListener(MainActivity.this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
                } else {
                    playPauseButton.setColor(Color.YELLOW);
                    senSensorManager.unregisterListener(MainActivity.this);
                    isRecording = false;
                    createChart();
                    String[] options = getResources().getStringArray(R.array.options);
                    //Shows options
                    AlertDialog.Builder builder = new AlertDialog.Builder(view.getContext());
                    builder.setTitle(R.string.what);
                    builder.setItems(options, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {

                            switch (which) {
                                case 0:

                                    //If send an email is selected then parse the data
                                    String route = Environment.getExternalStorageDirectory().getPath() + "/AccelerometerSleepWave";

                                    String title = "Sleep Wave Movement data";
                                    String FILENAME = "movement_data_gathering.xls";

                                    //Parse the data to xls
                                    XLS converter = new XLS();
                                    boolean isSucceed = converter.exportDataAsXLS(timestamps, x_axis, y_axis, z_axis, title,
                                            route, getResources().getStringArray(R.array.column_titles),
                                            view.getContext());

                                    //If parse is succeed
                                    if (isSucceed) {

                                        try {
                                            //Send an email with the file attached
                                            Intent sendIntent = new Intent(Intent.ACTION_SEND);
                                            sendIntent.setType("application/excel");
                                            sendIntent.putExtra(Intent.EXTRA_SUBJECT, title);
                                            sendIntent.putExtra(Intent.EXTRA_STREAM, Uri.fromFile(new File(route, FILENAME)));
                                            startActivity(Intent.createChooser(sendIntent, "Email:"));
                                        } catch (Exception e) {
                                            Toast.makeText(getBaseContext(), R.string.error_parsing_data, Toast.LENGTH_SHORT).show();
                                        }
                                    } else {
                                        Toast.makeText(getBaseContext(), R.string.error_parsing_data, Toast.LENGTH_SHORT).show();

                                    }
                                    break;

                                case 1:
                                    Toast.makeText(getBaseContext(), R.string.deleted, Toast.LENGTH_SHORT).show();
                                    break;
                            }
                        }
                    });
                    builder.show();
                }
            }
        });
    }

    private void initialitze() {
        timestamps = new ArrayList<>();
        x_axis = new ArrayList<>();
        y_axis = new ArrayList<>();
        z_axis = new ArrayList<>();
        lastUpdate = 0;
        isRecording = true;
    }

    protected void onPause() {
        super.onPause();
        if(isRecording)
            senSensorManager.unregisterListener(this);
    }

    protected void onResume() {
        super.onResume();
        if(isRecording)
            senSensorManager.registerListener(this, senAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        Sensor mySensor = event.sensor;

        //Checking again if it is the accelerometer
        if (mySensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            long curTime = System.currentTimeMillis();

            if ((curTime - lastUpdate) > 100) {
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];

                if(lastUpdate == 0) {
                    timestamps.add(lastUpdate);
                    lastUpdate = curTime;
                    firstTime = curTime;
                }
                else {
                    lastUpdate = curTime;
                    timestamps.add(lastUpdate - firstTime);
                }

                x_axis.add(x);
                y_axis.add(y);
                z_axis.add(z);
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    private void createChart() {
        Intent intent = new Intent(MainActivity.this, ChartActivity.class);
        intent.putExtra("x_data", x_axis);
        intent.putExtra("y_data", y_axis);
        intent.putExtra("z_data", z_axis);
        intent.putExtra("timestamps", timestamps);
        startActivity(intent);
    }
}
