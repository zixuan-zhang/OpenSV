
namespace cpp opensv

enum ErrorCode{
    ReferenceSignatureShortage,
    ReferenceSignatureQuality,
    AccountNotExist,
    TestSignatureNotFound,
    TooMuchTestSignatureGiven,
}

struct Point {
   1: i32 t,
   2: double x,
   3: double y,
   4: double p
}

struct Signature {
   1: list<Point> points,
}

struct Request {
   1: string id,
   2: list<Signature> signatures,
}

struct Ret {
   1: bool success,
   2: ErrorCode error, 
}

service HandWriter {
    i32 ping(1:i32 num),
    Ret accountRegister(1:Request request),
    Ret verify(1:Request request)
}
