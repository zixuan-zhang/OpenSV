package cn.ac.iscas.handwriter;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.util.Log;

import java.io.IOException;

import cn.ac.iscas.handwriter.utils.HandWriter;

/**
 * Created by zixuan on 2016/11/8.
 */
public class Database {
    private final String Tag = "Database";

    private final String DatabaseName = "handwriter.db3";

    private DatabaseHelper databaseHelper = null;

    private final String TableName = "signature_records";
    private final String TableSignatureIdColumn = "signatureId";
    private final String TableTimestampColumn = "timestamp";
    private final String TableXColumn = "x";
    private final String TableYColumn = "y";

    public Database(Context context) {
        databaseHelper = new DatabaseHelper(context, DatabaseName, 1);
    }

    public void insertData(int signatureId, double timestamp, double x, double y, double p) {
        ContentValues values = new ContentValues();
        values.put(TableSignatureIdColumn, signatureId);
        values.put(TableTimestampColumn, timestamp);
        values.put(TableXColumn, x);
        values.put(TableYColumn, y);
        databaseHelper.getReadableDatabase().insert(TableName, null, values);
    }

    public Cursor searchBySignatureId(int signatureId) {
        Cursor cursor = null;
        cursor = databaseHelper.getReadableDatabase().rawQuery(
                "select * from " + TableName + " where signatureId == " + signatureId + " order by " + TableTimestampColumn,
                null);
        return cursor;
    }

    public int searchTotalCount() {
        Cursor cursor = databaseHelper.getReadableDatabase().rawQuery(
                "select * from " + TableName, null);
        int count = cursor.getCount();
        Log.d(Tag, "searchTotalCount: " + count);
        try {
            HandWriter.sdLog.write("searchTotalCount: ".getBytes());
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return cursor.getCount();
    }

    /*
     * @Des: Delete all data in signature_record table
     */
    public boolean deleteFromSignatureRecord() {
        int count = databaseHelper.getWritableDatabase().delete(TableName, null, null);
        Log.d(Tag, "Delete from signature_records, affected rows: " + count);
        return count > 0;
    }
}
