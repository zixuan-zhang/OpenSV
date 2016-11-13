package cn.ac.iscas.handwriter;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import cn.ac.iscas.handwriter.utils.Config;

/**
 * Created by zixuan on 2016/11/7.
 */

/**
 * DatabaseHelper class.
 */
public class DatabaseHelper extends SQLiteOpenHelper {
    private final String Tag = "MyDatabaseHelper";
    private final String TableName = Config.DatabaseTableName;

    /**
     * Constructor of DatabaseHelper. Extends SQLiteOpenHelper.
     */
    public DatabaseHelper(Context context, String name, int version) {
        super(context, name, null, version);
    }

    /**
     * Once creation, this function will do two things.
     *     1. Create signature_records table.
     *     2. Make signatureId column as index.
     */
    @Override
    public void onCreate(SQLiteDatabase db) {
        String CreateSignatureRecordsSQL = "create table if not exists " + TableName + "(_id integer primary key autoincrement ," +
                "signatureId integer, timestamp real, x real, y real)";
        String CreateSignatureIdIndexSQL = "create index if not exists signatureId_index on " + TableName + " (signatureId ASC)";
        db.execSQL(CreateSignatureRecordsSQL);
        db.execSQL(CreateSignatureIdIndexSQL);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        Log.d(Tag, "update database from old version: " + oldVersion + " to new version: " + newVersion);
    }

    @Override
    public void onOpen(SQLiteDatabase db) {
        super.onOpen(db);
    }
}
