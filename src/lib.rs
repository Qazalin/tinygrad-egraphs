use egg::{rewrite as rw, *};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_pickle as pickle;
use std::os::raw::{c_char, c_int, c_uchar};
use std::slice;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
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

#[derive(Serialize, Deserialize, Debug)]
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
            let zero = UOp::const_(0);
            let val = UOp::const_(42);
            let x = UOp {
                op: UOps::ALU,
                dtype: Some("dtypes.int".to_string()),
                src: vec![zero, val],
                arg: Some(1), // this is "BinaryOps.ADD"
            };
            println!("{:?}", x);
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
