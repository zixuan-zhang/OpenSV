package cn.ac.iscas.handwriter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.os.Bundle;
import android.app.Activity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import cn.ac.iscas.handwriter.views.SignaturePad;
import cn.ac.iscas.handwriter.utils.HandWriter;
//add
import android.view.KeyEvent;
import android.view.WindowManager;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.SerializationHelper;
//end


public class UnlockActivity extends Activity {

    private final String TAG = "UnlockActivity";

    private SignaturePad mSignaturePad;
    private Button mClearButton;
    private Button mSaveButton;
    private TextView mSignaturepadDescription;

    //add
    private Button mOptionbtn;
    private int count;
    
    private static ArrayList mrecords=new ArrayList();

    private HandWriter mhandwriter;
    //end

    public static Database database = null;
    public static Classifier classifier = null;

    @Override
    public void onAttachedToWindow() {
		 this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN | 
		   WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD | 
		   WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED | 
		   WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON ,
		   WindowManager.LayoutParams.FLAG_FULLSCREEN | 
		   WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD | 
		   WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED | 
		   WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);
		 super.onAttachedToWindow();
	}

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AssetManager am = getAssets();
        String classifierDumpFile = "classifer.dump";
        try
        {
            InputStream classifierInput = am.open(classifierDumpFile);
            classifier = (RandomForest) SerializationHelper.read(classifierInput);
        }
        catch (IOException e)
        {
            System.out.println("IOException when load classifier dump file" + e.toString());
        }
        catch (Exception e)
        {
            System.out.println("Exception when load classifier dump file" + e.toString());
        }

        database = new Database(this);
        mhandwriter = HandWriter.GetInstance();

        mSignaturePad = (SignaturePad) findViewById(R.id.signature_pad);
        fullScreenDisplay();
        count=0;

        mSignaturePad.setOnSignedListener(new SignaturePad.OnSignedListener() {
            @Override
            public void onStartSigning() { //保留添加二级密码功能
                if (/*mCurrentUsername == null*/false) {
                    new AlertDialog.Builder(UnlockActivity.this)
                        .setTitle(R.string.no_username_dialog_title)
                        .setMessage(R.string.no_username_dialog_message)
                        .setIcon(android.R.drawable.stat_sys_warning)
                        .setCancelable(false)
                        .setPositiveButton("ok", new DialogInterface.OnClickListener() {
                                                     @Override
                                                     public void onClick(DialogInterface dialog, int which) {
                            mSignaturePad.clear();
                            // clear old records.
                            mSignaturePad.getMotionEventRecord().clear();
                            dialog.dismiss();
                        }
                    }).create().show();
                } else {
                    fullScreenDisplay();
                }
            }

            @Override
            public void onSigned() {
                mSaveButton.setEnabled(true);
                mClearButton.setEnabled(true);
                count=0;
            }

            @Override
            public void onClear() {
                mSaveButton.setEnabled(false);
                //mClearButton.setEnabled(false);
            }
        });

        mClearButton = (Button) findViewById(R.id.clear_button);
        mSaveButton = (Button) findViewById(R.id.save_button);
        mOptionbtn = (Button) findViewById(R.id.option_btn);
        mOptionbtn.setVisibility(View.INVISIBLE);

        mSignaturepadDescription = (TextView) findViewById(R.id.signature_pad_description);
        mSignaturepadDescription.setText(getString(R.string.hint_info3));

        mClearButton.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View view) {
            mSignaturePad.clear();
            // clear old records.
            mSignaturePad.getMotionEventRecord().clear();
            if(count==10)
            {
                Intent intent = new Intent("cn.ac.iscas.FACE_UNLOCK");
                sendBroadcast(intent);
                Toast.makeText(UnlockActivity.this, "解锁成功!", Toast.LENGTH_SHORT).show();
                finish();
            }
            count++;

        }
        });

        mSaveButton.setOnClickListener(new View.OnClickListener() {
                                           @Override
                                           public void onClick(View view) {
            // show dialog, let user tell us whether this signature is true.
            mrecords.clear();
            mrecords.add(mSignaturePad.getMotionEventRecord());
            mSignaturePad.clear();
            if (mhandwriter.check())
            {
                Intent intent = new Intent("cn.ac.iscas.FACE_UNLOCK");
                sendBroadcast(intent);
                Toast.makeText(UnlockActivity.this, "解锁成功!", Toast.LENGTH_SHORT).show();
                finish();
            }
            else
            {
                Toast.makeText(UnlockActivity.this, "解锁失败", Toast.LENGTH_SHORT).show();
            }
            count=0;
            mSignaturePad.clear();
            mSignaturePad.getMotionEventRecord().clear();
            fullScreenDisplay();
        }
        });
    }


    //选项按钮    
    private void fullScreenDisplay (){
    if (mSignaturePad != null) {
        // full screen setting, make our sign UI fullscreen.
        mSignaturePad.setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
    }
    }
    
    
    
    //add
    public static ArrayList getrecords(){
            return mrecords;
    }

    @Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		 
		if(KeyEvent.KEYCODE_BACK == event.getKeyCode())  
		{  
			return true; 
		}  
		return super.onKeyDown(keyCode, event); 
	}
    //end

}
