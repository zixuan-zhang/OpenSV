package cn.ac.iscas.handwriter;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

/**
 * Created by zixuan on 2016/11/7.
 */
public class DatabaseHelper extends SQLiteOpenHelper {
    public DatabaseHelper(Context context, String name, int version) {
        super(context, name, null, version);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        // TODO: Operations right after database creation.
        db.execSQL(CreateSignatureRecordsSQL);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // TODO: operations when version changed.
        Log.d(Tag, "update database from old version: " + oldVersion + " to new version: " + newVersion);
    }

    @Override
    public void onOpen(SQLiteDatabase db) {
        super.onOpen(db);
    }

    private final String Tag = "MyDatabaseHelper";

    private final String CreateSignatureRecordsSQL = "create table signature_records(_id integer primary key autoincrement ," +
            "signatureId integer, timestamp real, x real, y real)";
}

