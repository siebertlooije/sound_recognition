package com.mikelorbea.sleepwavetest;

import android.app.Activity;
import android.media.MediaRecorder;
import android.os.Environment;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.loopj.android.http.AsyncHttpClient;
import com.loopj.android.http.AsyncHttpResponseHandler;
import com.loopj.android.http.RequestParams;

import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import cz.msebera.android.httpclient.Header;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final Button start = (Button) findViewById(R.id.button);
        final Button stop = (Button) findViewById(R.id.stop);
        final TextView text = (TextView) findViewById(R.id.textView);

        final String url = "https://qtjspngw.p50.weaved.com/upload";

        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                text.setText("Recording file...");

                final String path = Environment.getExternalStorageDirectory().getPath();
                final MediaRecorder mRecorder = new MediaRecorder();
                mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                mRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
                mRecorder.setOutputFile(path + "/audio.mp3");
                mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);

                try {
                    mRecorder.prepare();
                } catch (IOException e) {
                    Log.e("Camera Error", "prepare() failed");
                }
                mRecorder.start();

                stop.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        mRecorder.stop();

                        text.setText("Uploading file...");

                        File myFile = new File(path + "/audio.mp3");
                        RequestParams params = new RequestParams();
                        String mobileID = "12345ABC";
                        try {
                            params.put(mobileID, myFile);
                        } catch(FileNotFoundException e) {}

                        // send request
                        AsyncHttpClient client = new AsyncHttpClient();
                        client.post(url, params, new AsyncHttpResponseHandler() {
                            @Override
                            public void onSuccess(int statusCode, Header[] headers, byte[] bytes) {
                                // handle success response
                                String str = new String(bytes);
                                text.setText("Success: " + str);
                                try {
                                    JSONObject json = new JSONObject(str);
                                }
                                catch (Exception e) {
                                    text.setText("Problem parsing JSON");
                                }
                            }
                            @Override
                            public void onFailure(int statusCode, Header[] headers, byte[] bytes, Throwable throwable) {
                                // handle failure response
                                text.setText("Failure: " + statusCode);
                            }
                        });
                    }
                });
            }
        });
    }
}
