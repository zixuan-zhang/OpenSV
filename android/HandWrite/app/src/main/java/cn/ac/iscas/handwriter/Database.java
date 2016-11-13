package cn.ac.iscas.handwriter;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.util.Log;

import cn.ac.iscas.handwriter.utils.Config;

/**
 * Created by zixuan on 2016/11/8.
 */

public class Database {
    private final String Tag = "Database";
    private final String DatabaseName = "handwriter.db3";
    private DatabaseHelper databaseHelper = null;
    private final String TableName = Config.DatabaseTableName;
    private final String TableSignatureIdColumn = "signatureId";
    private final String TableTimestampColumn = "timestamp";
    private final String TableXColumn = "x";
    private final String TableYColumn = "y";

    /**
     * Constructor of Database class. Initialize DatabaseHelper object.
     * @Param: context
     */
    public Database(Context context) {
        databaseHelper = new DatabaseHelper(context, DatabaseName, 1);
    }

    /**
     * Insert row into signature_records table. A row is a point record.
     * @Param: signatureId : ID of signature which this point belongs to.
     * @Param: timestamp : Timestamp of this point.
     * @Param: x : x axis of this point.
     * @Param: y : y axis of this point.
     * @Param: p : pressure of this point.
     * @Return: The row ID of the newly inserted row, or -1 if an error occurred.
     */
    public long insertData(int signatureId, double timestamp, double x, double y, double p) {
        ContentValues values = new ContentValues();
        values.put(TableSignatureIdColumn, signatureId);
        values.put(TableTimestampColumn, timestamp);
        values.put(TableXColumn, x);
        values.put(TableYColumn, y);
        return databaseHelper.getReadableDatabase().insert(TableName, null, values);
    }

    /**
     * Search signature_records by signatureId column.
     * @Param: signatureId : ID of signature which this point belongs to.
     * @Return: Database cursor.
     */
    public Cursor searchBySignatureId(int signatureId) {
        Cursor cursor = null;
        cursor = databaseHelper.getReadableDatabase().rawQuery(
                "select * from " + TableName + " where signatureId == " + signatureId + " order by " + TableTimestampColumn,
                null);
        return cursor;
    }

    /**
     * Search total row count in signature_records table.
     * @Return: The total row count.
     */
    public int searchTotalCount() {
        Cursor cursor = databaseHelper.getReadableDatabase().rawQuery(
                "select * from " + TableName, null);
        int count = cursor.getCount();
        Log.d(Tag, "searchTotalCount: " + count);
        return cursor.getCount();
    }

    /**
     * Delete all data in signature_record table
     * @Return: true for success else false.
     */
    public boolean deleteFromSignatureRecord() {
        int count = databaseHelper.getWritableDatabase().delete(TableName, null, null);
        Log.d(Tag, "Delete from signature_records, affected rows: " + count);
        return count > 0;
    }
}
