package cn.ac.iscas.handwriter;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.app.Activity;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.util.SparseArray;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.PopupMenu;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;

import cn.ac.iscas.handwriter.views.SignaturePad;
import cn.ac.iscas.handwriter.utils.HandWriter;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.SerializationHelper;

public class MainActivity extends Activity {

    private final String TAG = "MainActivity";

    private SignaturePad mSignaturePad;
    private Button mClearButton;
    private Button mSaveButton;
    private TextView mSignaturepadDescription;

    private PopupMenu popupMenu = null;

    // shared prefs
    private final String PREFS_USER = "cn.ac.iscas.handwriter.currentuser";
    private SharedPreferences mUserPreferences;
    private SharedPreferences.Editor mUserEditor;

    //add
    private boolean msyslock;
    private boolean mapplock;
    private final String SYSLCK_KEY = "SYSTEMLOCK";
    private final String APPLCK_KEY = "APPLOCK";
    
    private static ArrayList mrecords=new ArrayList();
    private HandWriter mhandwriter= HandWriter.GetInstance();
    //end

    public static Classifier classifier = null;

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


        mUserPreferences = getSharedPreferences(PREFS_USER, MODE_PRIVATE);
        mUserEditor = mUserPreferences.edit();

        msyslock=mUserPreferences.getBoolean(SYSLCK_KEY,false);


        mSignaturePad = (SignaturePad) findViewById(R.id.signature_pad);
        fullScreenDisplay();

        mSignaturePad.setOnSignedListener(new SignaturePad.OnSignedListener() {
            @Override
            public void onStartSigning() { //保留添加二级密码功能
            if (/*mCurrentUsername == null*/false) {
                new AlertDialog.Builder(MainActivity.this)
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
    }

        @Override
        public void onClear() {
        mSaveButton.setEnabled(false);
        mClearButton.setEnabled(false);
        }
    });

    mClearButton = (Button) findViewById(R.id.clear_button);
    mSaveButton = (Button) findViewById(R.id.save_button);
    mSignaturepadDescription = (TextView) findViewById(R.id.signature_pad_description);

    mClearButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View view) {
        mSignaturePad.clear();
        // clear old records.
        mSignaturePad.getMotionEventRecord().clear();
        }
    });

    mSaveButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View view) {
        // show dialog, let user tell us whether this signature is true.
        if(mrecords.size()==5)
            mrecords.clear();
        mrecords.add(mSignaturePad.getMotionEventRecord().clone());
        register(0);
        mSignaturePad.clear();
        mSignaturePad.getMotionEventRecord().clear();
        mSignaturepadDescription.setText(getString(R.string.hint_info) + mrecords.size()+getString(R.string.hint_info2));
        fullScreenDisplay();
        }
    });
    
    mSignaturepadDescription.setText(getString(R.string.hint_info) + mrecords.size()+getString(R.string.hint_info2));

    }

    //选项按钮    
    public void onOptionClicked(View view) {

    if (popupMenu == null) {
        popupMenu = new PopupMenu(this, view);
       
        getMenuInflater().inflate(R.menu.popup_menu, popupMenu.getMenu());

        popupMenu.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
        @Override
        public boolean onMenuItemClick(final MenuItem item) {

            switch (item.getItemId()) {
            case R.id.new_user:
                mrecords.clear();
                mSignaturePad.clear();
                mSignaturePad.getMotionEventRecord().clear();
                mSignaturepadDescription.setText(getString(R.string.hint_info) + mrecords.size()+getString(R.string.hint_info2));
                fullScreenDisplay();
            break;

            //add
            case R.id.check_sys:
                if(msyslock)
                {
                    msyslock=false;
                    mUserEditor.putBoolean(SYSLCK_KEY,msyslock);
                    mUserEditor.commit();
                    Toast.makeText(MainActivity.this, "关闭系统锁!", Toast.LENGTH_SHORT).show();
                }
                else
                {
                    register(1);
                }
                item.setChecked(msyslock);
            break;

            case R.id.check_app:
                Toast.makeText(MainActivity.this, "敬请期待!", Toast.LENGTH_SHORT).show();
                Intent intentLockAppActivity = new Intent(MainActivity.this,UnlockActivity.class);
        	        intentLockAppActivity.putExtra(Intent.EXTRA_SHORTCUT_NAME, "ScreenLOCK");
                intentLockAppActivity.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK  );
			    startActivity(intentLockAppActivity);
            break;
            //add
            default:
                setPaintColor(item);
            break;
            }

            return true;
        }
        });

    }

    // Reflect to invoke setForceShowIcon function to show menu icon.
    // we may get IllegalAccessException: access to field not allowed here,
    // but it's ok, we just catch it and ignore it.
    try {
        Field[] fields = popupMenu.getClass().getDeclaredFields();
        for (Field field : fields) {
        if ("mPopup".equals(field.getName())) {
            field.setAccessible(true);
            Object menuPopupHelper = field.get(popupMenu);
            Class<?> classPopupHelper = Class.forName(menuPopupHelper
                .getClass().getName());
            Method setForceIcons = classPopupHelper.getMethod(
                "setForceShowIcon", boolean.class);
            setForceIcons.invoke(menuPopupHelper, true);
            break;
        }
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    MenuItem item=popupMenu.getMenu().findItem(R.id.check_sys);
    item.setChecked(msyslock);
    popupMenu.show();
    }
    

    private void setPaintColor(MenuItem item) {
    item.setChecked(true);
    switch (item.getItemId()) {
        case R.id.paint_color_black:
        mSignaturePad.setPaintColor(Color.BLACK);
        break;
        case R.id.paint_color_blue:
        mSignaturePad.setPaintColor(Color.BLUE);
        break;
        case R.id.paint_color_cyan:
        mSignaturePad.setPaintColor(Color.CYAN);
        break;
        case R.id.paint_color_dkgray:
        mSignaturePad.setPaintColor(Color.DKGRAY);
        break;
        case R.id.paint_color_gray:
        mSignaturePad.setPaintColor(Color.GRAY);
        break;
        case R.id.paint_color_green:
        mSignaturePad.setPaintColor(Color.GREEN);
        break;
        case R.id.paint_color_ltgray:
        mSignaturePad.setPaintColor(Color.LTGRAY);
        break;
        case R.id.paint_color_magenta:
        mSignaturePad.setPaintColor(Color.MAGENTA);
        break;
        case R.id.paint_color_red:
        mSignaturePad.setPaintColor(Color.RED);
        break;
        case R.id.paint_color_yellow:
        mSignaturePad.setPaintColor(Color.YELLOW);
        break;
    }
    fullScreenDisplay();
    }


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

    private void register(int type)
    {
        if(mrecords.size()==5)
           {
            boolean temp=mhandwriter.register();
            if (temp)
            {
                Toast.makeText(MainActivity.this, "注册成功", Toast.LENGTH_SHORT).show();
                msyslock=true;
                mUserEditor.putBoolean(SYSLCK_KEY,msyslock);
                mUserEditor.commit();
            }
            else
            {
                Toast.makeText(MainActivity.this, "注册失败", Toast.LENGTH_SHORT).show();
                mrecords.clear();
                msyslock=false;
                mUserEditor.putBoolean(SYSLCK_KEY,msyslock);
                mUserEditor.commit();
            }
        }
        else {
            if(type==1)
                Toast.makeText(MainActivity.this, "签名个数不足", Toast.LENGTH_SHORT).show();
        }
    }
    //end

}
