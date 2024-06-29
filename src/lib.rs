#[allow(unused)]
use egg::{rewrite as rw, *};
use serde::{Deserialize, Serialize};
use serde_pickle as pickle;
use std::os::raw::{c_char, c_int, c_uchar};
use std::slice;

// *** ops
#[derive(Debug)]
enum UnaryOps {
    /// A -> A (elementwise)
    EXP2,
    LOG2,
    CAST,
    BITCAST,
    SIN,
    SQRT,
    NEG,
    RECIP,
}
#[derive(Debug)]
enum BinaryOps {
    /// A + A -> A (elementwise)
    ADD,
    MUL,
    IDIV,
    MAX,
    MOD,
    CMPLT,
    CMPNE,
    XOR,
    SHR,
    SHL,
}
#[derive(Debug)]
enum TernaryOps {
    /// A + A + A -> A (elementwise)
    WHERE,
    MULACC,
}
#[derive(Debug)]
enum ReduceOps {
    /// A -> B (reduce)
    SUM,
    MAX,
}
#[derive(Debug)]
enum BufferOps {
    LOAD,
    CONST,
    STORE,
}

// *** uops
#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
enum UOps {
    SINK,
    VAR,
    DEFINE_GLOBAL,
    DEFINE_VAR,
    DEFINE_LOCAL,
    DEFINE_ACC,
    CONST,
    SPECIAL,
    NOOP,
    UNMUL,
    GEP,
    // math ops
    CAST,
    BITCAST,
    ALU,
    WMMA,
    // memory/assignment ops
    LOAD,
    STORE,
    PHI,
    // control flow ops
    BARRIER,
    IF,
    RANGE,
    // these two are not graph nodes
    ENDRANGE,
    ENDIF,
}
#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
struct UOp {
    op: UOps,
    dtype: Option<String>,
    src: Vec<UOp>,
    arg: Option<u32>,
}
impl UOp {
    fn const_(x: u32) -> UOp {
        return UOp {
            op: UOps::CONST,
            dtype: Some("dtypes.int".to_string()),
            src: vec![],
            arg: Some(x),
        };
    }
}

// *** api
#[repr(C)]
pub struct ByteArray {
    ptr: *mut c_char,
    len: usize,
}

#[no_mangle]
pub extern "C" fn rewrite_uops(data: *const c_uchar, len: c_int) -> ByteArray {
    let bytes = unsafe { slice::from_raw_parts(data, len as usize) };
    let uop = pickle::from_slice::<UOp>(bytes, serde_pickle::DeOptions::new());
    match uop {
        Ok(_) => {
            let new = UOp {
                op: UOps::CONST,
                dtype: Some("dtypes.int".to_string()),
                src: vec![],
                arg: Some(42),
            };
            let ret = serde_pickle::to_vec(&new, serde_pickle::SerOptions::new()).unwrap();
            let len = ret.len();
            let ptr = ret.as_ptr() as *mut c_char;
            std::mem::forget(ret);
            ByteArray { ptr, len }
        }
        Err(err) => {
            let value =
                pickle::from_slice::<serde_pickle::Value>(bytes, serde_pickle::DeOptions::new())
                    .unwrap();
            panic!("couldn't handle {:?}\n error = {:?}", value, err);
        }
    }
}

#[cfg(test)]
mod test_tiny {
    use super::*;
    #[test]
    fn test_uop_eq() {
        let u0 = UOp::const_(0);
        let u1 = UOp::const_(0);
        assert_eq!(u0, u1)
    }
}
