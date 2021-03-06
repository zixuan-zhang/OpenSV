/**
 * Autogenerated by Thrift Compiler (0.9.2)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
import org.apache.thrift.scheme.IScheme;
import org.apache.thrift.scheme.SchemeFactory;
import org.apache.thrift.scheme.StandardScheme;

import org.apache.thrift.scheme.TupleScheme;
import org.apache.thrift.protocol.TTupleProtocol;
import org.apache.thrift.protocol.TProtocolException;
import org.apache.thrift.EncodingUtils;
import org.apache.thrift.TException;
import org.apache.thrift.async.AsyncMethodCallback;
import org.apache.thrift.server.AbstractNonblockingServer.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.EnumMap;
import java.util.Set;
import java.util.HashSet;
import java.util.EnumSet;
import java.util.Collections;
import java.util.BitSet;
import java.nio.ByteBuffer;
import java.util.Arrays;
import javax.annotation.Generated;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings({"cast", "rawtypes", "serial", "unchecked"})
@Generated(value = "Autogenerated by Thrift Compiler (0.9.2)", date = "2017-3-19")
public class Point implements org.apache.thrift.TBase<Point, Point._Fields>, java.io.Serializable, Cloneable, Comparable<Point> {
  private static final org.apache.thrift.protocol.TStruct STRUCT_DESC = new org.apache.thrift.protocol.TStruct("Point");

  private static final org.apache.thrift.protocol.TField T_FIELD_DESC = new org.apache.thrift.protocol.TField("t", org.apache.thrift.protocol.TType.I32, (short)1);
  private static final org.apache.thrift.protocol.TField X_FIELD_DESC = new org.apache.thrift.protocol.TField("x", org.apache.thrift.protocol.TType.DOUBLE, (short)2);
  private static final org.apache.thrift.protocol.TField Y_FIELD_DESC = new org.apache.thrift.protocol.TField("y", org.apache.thrift.protocol.TType.DOUBLE, (short)3);
  private static final org.apache.thrift.protocol.TField P_FIELD_DESC = new org.apache.thrift.protocol.TField("p", org.apache.thrift.protocol.TType.DOUBLE, (short)4);

  private static final Map<Class<? extends IScheme>, SchemeFactory> schemes = new HashMap<Class<? extends IScheme>, SchemeFactory>();
  static {
    schemes.put(StandardScheme.class, new PointStandardSchemeFactory());
    schemes.put(TupleScheme.class, new PointTupleSchemeFactory());
  }

  public int t; // required
  public double x; // required
  public double y; // required
  public double p; // required

  /** The set of fields this struct contains, along with convenience methods for finding and manipulating them. */
  public enum _Fields implements org.apache.thrift.TFieldIdEnum {
    T((short)1, "t"),
    X((short)2, "x"),
    Y((short)3, "y"),
    P((short)4, "p");

    private static final Map<String, _Fields> byName = new HashMap<String, _Fields>();

    static {
      for (_Fields field : EnumSet.allOf(_Fields.class)) {
        byName.put(field.getFieldName(), field);
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, or null if its not found.
     */
    public static _Fields findByThriftId(int fieldId) {
      switch(fieldId) {
        case 1: // T
          return T;
        case 2: // X
          return X;
        case 3: // Y
          return Y;
        case 4: // P
          return P;
        default:
          return null;
      }
    }

    /**
     * Find the _Fields constant that matches fieldId, throwing an exception
     * if it is not found.
     */
    public static _Fields findByThriftIdOrThrow(int fieldId) {
      _Fields fields = findByThriftId(fieldId);
      if (fields == null) throw new IllegalArgumentException("Field " + fieldId + " doesn't exist!");
      return fields;
    }

    /**
     * Find the _Fields constant that matches name, or null if its not found.
     */
    public static _Fields findByName(String name) {
      return byName.get(name);
    }

    private final short _thriftId;
    private final String _fieldName;

    _Fields(short thriftId, String fieldName) {
      _thriftId = thriftId;
      _fieldName = fieldName;
    }

    public short getThriftFieldId() {
      return _thriftId;
    }

    public String getFieldName() {
      return _fieldName;
    }
  }

  // isset id assignments
  private static final int __T_ISSET_ID = 0;
  private static final int __X_ISSET_ID = 1;
  private static final int __Y_ISSET_ID = 2;
  private static final int __P_ISSET_ID = 3;
  private byte __isset_bitfield = 0;
  public static final Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> metaDataMap;
  static {
    Map<_Fields, org.apache.thrift.meta_data.FieldMetaData> tmpMap = new EnumMap<_Fields, org.apache.thrift.meta_data.FieldMetaData>(_Fields.class);
    tmpMap.put(_Fields.T, new org.apache.thrift.meta_data.FieldMetaData("t", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.I32)));
    tmpMap.put(_Fields.X, new org.apache.thrift.meta_data.FieldMetaData("x", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    tmpMap.put(_Fields.Y, new org.apache.thrift.meta_data.FieldMetaData("y", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    tmpMap.put(_Fields.P, new org.apache.thrift.meta_data.FieldMetaData("p", org.apache.thrift.TFieldRequirementType.DEFAULT, 
        new org.apache.thrift.meta_data.FieldValueMetaData(org.apache.thrift.protocol.TType.DOUBLE)));
    metaDataMap = Collections.unmodifiableMap(tmpMap);
    org.apache.thrift.meta_data.FieldMetaData.addStructMetaDataMap(Point.class, metaDataMap);
  }

  public Point() {
  }

  public Point(
    int t,
    double x,
    double y,
    double p)
  {
    this();
    this.t = t;
    setTIsSet(true);
    this.x = x;
    setXIsSet(true);
    this.y = y;
    setYIsSet(true);
    this.p = p;
    setPIsSet(true);
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public Point(Point other) {
    __isset_bitfield = other.__isset_bitfield;
    this.t = other.t;
    this.x = other.x;
    this.y = other.y;
    this.p = other.p;
  }

  public Point deepCopy() {
    return new Point(this);
  }

  @Override
  public void clear() {
    setTIsSet(false);
    this.t = 0;
    setXIsSet(false);
    this.x = 0.0;
    setYIsSet(false);
    this.y = 0.0;
    setPIsSet(false);
    this.p = 0.0;
  }

  public int getT() {
    return this.t;
  }

  public Point setT(int t) {
    this.t = t;
    setTIsSet(true);
    return this;
  }

  public void unsetT() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __T_ISSET_ID);
  }

  /** Returns true if field t is set (has been assigned a value) and false otherwise */
  public boolean isSetT() {
    return EncodingUtils.testBit(__isset_bitfield, __T_ISSET_ID);
  }

  public void setTIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __T_ISSET_ID, value);
  }

  public double getX() {
    return this.x;
  }

  public Point setX(double x) {
    this.x = x;
    setXIsSet(true);
    return this;
  }

  public void unsetX() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __X_ISSET_ID);
  }

  /** Returns true if field x is set (has been assigned a value) and false otherwise */
  public boolean isSetX() {
    return EncodingUtils.testBit(__isset_bitfield, __X_ISSET_ID);
  }

  public void setXIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __X_ISSET_ID, value);
  }

  public double getY() {
    return this.y;
  }

  public Point setY(double y) {
    this.y = y;
    setYIsSet(true);
    return this;
  }

  public void unsetY() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __Y_ISSET_ID);
  }

  /** Returns true if field y is set (has been assigned a value) and false otherwise */
  public boolean isSetY() {
    return EncodingUtils.testBit(__isset_bitfield, __Y_ISSET_ID);
  }

  public void setYIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __Y_ISSET_ID, value);
  }

  public double getP() {
    return this.p;
  }

  public Point setP(double p) {
    this.p = p;
    setPIsSet(true);
    return this;
  }

  public void unsetP() {
    __isset_bitfield = EncodingUtils.clearBit(__isset_bitfield, __P_ISSET_ID);
  }

  /** Returns true if field p is set (has been assigned a value) and false otherwise */
  public boolean isSetP() {
    return EncodingUtils.testBit(__isset_bitfield, __P_ISSET_ID);
  }

  public void setPIsSet(boolean value) {
    __isset_bitfield = EncodingUtils.setBit(__isset_bitfield, __P_ISSET_ID, value);
  }

  public void setFieldValue(_Fields field, Object value) {
    switch (field) {
    case T:
      if (value == null) {
        unsetT();
      } else {
        setT((Integer)value);
      }
      break;

    case X:
      if (value == null) {
        unsetX();
      } else {
        setX((Double)value);
      }
      break;

    case Y:
      if (value == null) {
        unsetY();
      } else {
        setY((Double)value);
      }
      break;

    case P:
      if (value == null) {
        unsetP();
      } else {
        setP((Double)value);
      }
      break;

    }
  }

  public Object getFieldValue(_Fields field) {
    switch (field) {
    case T:
      return Integer.valueOf(getT());

    case X:
      return Double.valueOf(getX());

    case Y:
      return Double.valueOf(getY());

    case P:
      return Double.valueOf(getP());

    }
    throw new IllegalStateException();
  }

  /** Returns true if field corresponding to fieldID is set (has been assigned a value) and false otherwise */
  public boolean isSet(_Fields field) {
    if (field == null) {
      throw new IllegalArgumentException();
    }

    switch (field) {
    case T:
      return isSetT();
    case X:
      return isSetX();
    case Y:
      return isSetY();
    case P:
      return isSetP();
    }
    throw new IllegalStateException();
  }

  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof Point)
      return this.equals((Point)that);
    return false;
  }

  public boolean equals(Point that) {
    if (that == null)
      return false;

    boolean this_present_t = true;
    boolean that_present_t = true;
    if (this_present_t || that_present_t) {
      if (!(this_present_t && that_present_t))
        return false;
      if (this.t != that.t)
        return false;
    }

    boolean this_present_x = true;
    boolean that_present_x = true;
    if (this_present_x || that_present_x) {
      if (!(this_present_x && that_present_x))
        return false;
      if (this.x != that.x)
        return false;
    }

    boolean this_present_y = true;
    boolean that_present_y = true;
    if (this_present_y || that_present_y) {
      if (!(this_present_y && that_present_y))
        return false;
      if (this.y != that.y)
        return false;
    }

    boolean this_present_p = true;
    boolean that_present_p = true;
    if (this_present_p || that_present_p) {
      if (!(this_present_p && that_present_p))
        return false;
      if (this.p != that.p)
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    List<Object> list = new ArrayList<Object>();

    boolean present_t = true;
    list.add(present_t);
    if (present_t)
      list.add(t);

    boolean present_x = true;
    list.add(present_x);
    if (present_x)
      list.add(x);

    boolean present_y = true;
    list.add(present_y);
    if (present_y)
      list.add(y);

    boolean present_p = true;
    list.add(present_p);
    if (present_p)
      list.add(p);

    return list.hashCode();
  }

  @Override
  public int compareTo(Point other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;

    lastComparison = Boolean.valueOf(isSetT()).compareTo(other.isSetT());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetT()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.t, other.t);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetX()).compareTo(other.isSetX());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetX()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.x, other.x);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetY()).compareTo(other.isSetY());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetY()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.y, other.y);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    lastComparison = Boolean.valueOf(isSetP()).compareTo(other.isSetP());
    if (lastComparison != 0) {
      return lastComparison;
    }
    if (isSetP()) {
      lastComparison = org.apache.thrift.TBaseHelper.compareTo(this.p, other.p);
      if (lastComparison != 0) {
        return lastComparison;
      }
    }
    return 0;
  }

  public _Fields fieldForId(int fieldId) {
    return _Fields.findByThriftId(fieldId);
  }

  public void read(org.apache.thrift.protocol.TProtocol iprot) throws org.apache.thrift.TException {
    schemes.get(iprot.getScheme()).getScheme().read(iprot, this);
  }

  public void write(org.apache.thrift.protocol.TProtocol oprot) throws org.apache.thrift.TException {
    schemes.get(oprot.getScheme()).getScheme().write(oprot, this);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Point(");
    boolean first = true;

    sb.append("t:");
    sb.append(this.t);
    first = false;
    if (!first) sb.append(", ");
    sb.append("x:");
    sb.append(this.x);
    first = false;
    if (!first) sb.append(", ");
    sb.append("y:");
    sb.append(this.y);
    first = false;
    if (!first) sb.append(", ");
    sb.append("p:");
    sb.append(this.p);
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws org.apache.thrift.TException {
    // check for required fields
    // check for sub-struct validity
  }

  private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
    try {
      write(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(out)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
    try {
      // it doesn't seem like you should have to do this, but java serialization is wacky, and doesn't call the default constructor.
      __isset_bitfield = 0;
      read(new org.apache.thrift.protocol.TCompactProtocol(new org.apache.thrift.transport.TIOStreamTransport(in)));
    } catch (org.apache.thrift.TException te) {
      throw new java.io.IOException(te);
    }
  }

  private static class PointStandardSchemeFactory implements SchemeFactory {
    public PointStandardScheme getScheme() {
      return new PointStandardScheme();
    }
  }

  private static class PointStandardScheme extends StandardScheme<Point> {

    public void read(org.apache.thrift.protocol.TProtocol iprot, Point struct) throws org.apache.thrift.TException {
      org.apache.thrift.protocol.TField schemeField;
      iprot.readStructBegin();
      while (true)
      {
        schemeField = iprot.readFieldBegin();
        if (schemeField.type == org.apache.thrift.protocol.TType.STOP) { 
          break;
        }
        switch (schemeField.id) {
          case 1: // T
            if (schemeField.type == org.apache.thrift.protocol.TType.I32) {
              struct.t = iprot.readI32();
              struct.setTIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 2: // X
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.x = iprot.readDouble();
              struct.setXIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 3: // Y
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.y = iprot.readDouble();
              struct.setYIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          case 4: // P
            if (schemeField.type == org.apache.thrift.protocol.TType.DOUBLE) {
              struct.p = iprot.readDouble();
              struct.setPIsSet(true);
            } else { 
              org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
            }
            break;
          default:
            org.apache.thrift.protocol.TProtocolUtil.skip(iprot, schemeField.type);
        }
        iprot.readFieldEnd();
      }
      iprot.readStructEnd();

      // check for required fields of primitive type, which can't be checked in the validate method
      struct.validate();
    }

    public void write(org.apache.thrift.protocol.TProtocol oprot, Point struct) throws org.apache.thrift.TException {
      struct.validate();

      oprot.writeStructBegin(STRUCT_DESC);
      oprot.writeFieldBegin(T_FIELD_DESC);
      oprot.writeI32(struct.t);
      oprot.writeFieldEnd();
      oprot.writeFieldBegin(X_FIELD_DESC);
      oprot.writeDouble(struct.x);
      oprot.writeFieldEnd();
      oprot.writeFieldBegin(Y_FIELD_DESC);
      oprot.writeDouble(struct.y);
      oprot.writeFieldEnd();
      oprot.writeFieldBegin(P_FIELD_DESC);
      oprot.writeDouble(struct.p);
      oprot.writeFieldEnd();
      oprot.writeFieldStop();
      oprot.writeStructEnd();
    }

  }

  private static class PointTupleSchemeFactory implements SchemeFactory {
    public PointTupleScheme getScheme() {
      return new PointTupleScheme();
    }
  }

  private static class PointTupleScheme extends TupleScheme<Point> {

    @Override
    public void write(org.apache.thrift.protocol.TProtocol prot, Point struct) throws org.apache.thrift.TException {
      TTupleProtocol oprot = (TTupleProtocol) prot;
      BitSet optionals = new BitSet();
      if (struct.isSetT()) {
        optionals.set(0);
      }
      if (struct.isSetX()) {
        optionals.set(1);
      }
      if (struct.isSetY()) {
        optionals.set(2);
      }
      if (struct.isSetP()) {
        optionals.set(3);
      }
      oprot.writeBitSet(optionals, 4);
      if (struct.isSetT()) {
        oprot.writeI32(struct.t);
      }
      if (struct.isSetX()) {
        oprot.writeDouble(struct.x);
      }
      if (struct.isSetY()) {
        oprot.writeDouble(struct.y);
      }
      if (struct.isSetP()) {
        oprot.writeDouble(struct.p);
      }
    }

    @Override
    public void read(org.apache.thrift.protocol.TProtocol prot, Point struct) throws org.apache.thrift.TException {
      TTupleProtocol iprot = (TTupleProtocol) prot;
      BitSet incoming = iprot.readBitSet(4);
      if (incoming.get(0)) {
        struct.t = iprot.readI32();
        struct.setTIsSet(true);
      }
      if (incoming.get(1)) {
        struct.x = iprot.readDouble();
        struct.setXIsSet(true);
      }
      if (incoming.get(2)) {
        struct.y = iprot.readDouble();
        struct.setYIsSet(true);
      }
      if (incoming.get(3)) {
        struct.p = iprot.readDouble();
        struct.setPIsSet(true);
      }
    }
  }

}

