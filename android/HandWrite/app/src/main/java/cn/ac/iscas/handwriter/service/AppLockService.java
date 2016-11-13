package cn.ac.iscas.handwriter.service; 
  
import cn.ac.iscas.handwriter.UnlockActivity;
import android.app.KeyguardManager;
import android.app.Service;
import android.content.BroadcastReceiver;  
import android.content.Context;  
import android.content.Intent;  
import android.content.IntentFilter;
import android.os.IBinder;
import android.util.Log;  


public class AppLockService extends Service{  
    private static String TAG = "ScreenObserver";
    public static boolean hasoffed = true;
    private Context mContext;  
    private ScreenBroadcastReceiver mScreenReceiver;  
 
    //add
    private final String PREFS_USER = "cn.ac.iscas.handwriter.currentuser";
    private final String SYSLCK_KEY = "SYSTEMLOCK";
    //end

    private Intent intentLockAppActivity;
    public AppLockService(){  
        mContext = this ;  
        mScreenReceiver = new ScreenBroadcastReceiver();    
    }  
  
    private class ScreenBroadcastReceiver extends BroadcastReceiver{
        private String action = null;
        @Override
        public void onReceive(Context context, Intent intent) {
            action = intent.getAction();
            if(Intent.ACTION_SCREEN_ON.equals(action)){
                 onScreenOn();
            }else if(Intent.ACTION_SCREEN_OFF.equals(action)){
                 onScreenOff();
            }
            else if (Intent.ACTION_USER_PRESENT.equals(action)) {
            	onUserPresent();
            }
        }
    }
    public void onCreate( ) {
    	
    	super.onCreate();   	 
        Log.d(TAG,"START SERVICE");
        startScreenBroadcastReceiver();
        onScreenOn();  
    }

	public void stopScreenStateUpdate(){
        mContext.unregisterReceiver(mScreenReceiver);  
    }  

    private void startScreenBroadcastReceiver(){
        IntentFilter filter = new IntentFilter();
        filter.addAction(Intent.ACTION_SCREEN_ON);
        filter.addAction(Intent.ACTION_SCREEN_OFF);
        filter.addAction(Intent.ACTION_USER_PRESENT);

        mContext.registerReceiver(mScreenReceiver, filter);
    }

	public void onScreenOn()
    {
    	Log.d(TAG, "onScreenOn");

    	boolean SceenLockstate = getSharedPreferences(PREFS_USER, MODE_PRIVATE).getBoolean(SYSLCK_KEY,false) ;

    	KeyguardManager keyguardManager = (KeyguardManager) getSystemService(Context.KEYGUARD_SERVICE);
    	boolean devicelocked = keyguardManager.isKeyguardSecure();
    	if( hasoffed && devicelocked && SceenLockstate)
    	{
    		hasoffed = false;
    		intentLockAppActivity = new Intent(this,UnlockActivity.class);
	    	intentLockAppActivity.putExtra(Intent.EXTRA_SHORTCUT_NAME, "ScreenLOCK");
		intentLockAppActivity.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK  );
		startActivity(intentLockAppActivity);
    	}

    }

	public void onUserPresent()
    {
    	Log.d(TAG, "onUserPresent");
    	hasoffed = false;
    }

	public void onScreenOff()
    {
    	Log.d(TAG, "onScreenOff");
    	hasoffed = true;
    }

	@Override
	public IBinder onBind(Intent intent) {
		return null;
	}
}  
