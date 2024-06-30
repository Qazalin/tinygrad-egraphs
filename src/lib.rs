#[allow(unused)]
use egg::{rewrite as rw, *};
use num_enum::{FromPrimitive, IntoPrimitive};
use serde::{Deserialize, Serialize};
use serde_pickle as pickle;
use std::os::raw::{c_char, c_int, c_uchar};
use std::{convert::Infallible, env, fmt::Display, slice};

lazy_static::lazy_static! {
    pub static ref GRAPH: bool = env::var("GRAPH").map(|v| v == "1").unwrap_or(false);
}

// *** ops
#[derive(Debug, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
enum UnaryOps {
    /// A -> A (elementwise)
    #[num_enum(default)]
    EXP2,
    LOG2,
    CAST,
    BITCAST,
    SIN,
    SQRT,
    NEG,
    RECIP,
}
#[derive(Debug, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
enum BinaryOps {
    /// A + A -> A (elementwise)
    #[num_enum(default)]
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
#[derive(Debug, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
enum TernaryOps {
    /// A + A + A -> A (elementwise)
    #[num_enum(default)]
    WHERE,
    MULACC,
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
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
struct UOp {
    op: UOps,
    dtype: Option<String>,
    src: Vec<UOp>,
    arg: Option<String>,
}

impl UOp {
    fn alu<T: Into<u32>, D: ToString>(op: T, src: Vec<UOp>) -> UOp {
        let op: u32 = op.into();
        return UOp {
            op: UOps::ALU,
            dtype: Some(src.last().unwrap().dtype.clone().unwrap()),
            src,
            arg: Some(op.to_string()),
        };
    }
}

impl<T> From<T> for UOp
where
    T: num::Num + ToString,
{
    fn from(value: T) -> Self {
        UOp {
            op: UOps::CONST,
            dtype: Some("dtypes.int".into()),
            src: vec![],
            arg: Some(value.to_string()),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
struct UOpLang {
    op: UOps,
    dtype: Option<String>,
    src: Vec<Id>,
    arg: Option<String>,
}

impl Display for UOpLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {} {} {:?}",
            self.op,
            self.dtype.clone().unwrap_or("".into()),
            self.arg.clone().unwrap_or("".into()),
            self.src
        )
    }
}

impl Language for UOpLang {
    fn matches(&self, other: &Self) -> bool {
        self.op == other.op && self.len() == other.len()
    }

    fn children(&self) -> &[Id] {
        &self.src
    }

    fn children_mut(&mut self) -> &mut [Id] {
        &mut self.src
    }
}

impl FromOp for UOpLang {
    type Error = Infallible;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        if let Some(const_literal) = op.parse::<u32>().ok() {
            return Ok(Self {
                op: UOps::CONST,
                dtype: Some("dtypes.int".to_string()),
                src: vec![],
                arg: Some(const_literal.to_string()),
            });
        }

        let expr = match op {
            "+" => Self {
                op: UOps::ALU,
                dtype: Some("dtypes.int".to_string()),
                src: children,
                arg: Some((BinaryOps::ADD as u32).to_string()),
            },
            "*" => Self {
                op: UOps::ALU,
                dtype: Some("dtypes.int".to_string()),
                src: children,
                arg: Some((BinaryOps::MUL as u32).to_string()),
            },
            "load" => Self {
                op: UOps::LOAD,
                dtype: Some("dtypes.int".to_string()),
                src: children,
                arg: None,
            },
            _ => todo!("{op}"),
        };
        Ok(expr)
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
        Ok(uop) => {
            println!("{:?}", uop);
            let new: UOp = 42.into();
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

#[derive(Debug)]
struct UOpEGraph {
    egraph: EGraph<UOpLang, ()>,
    sink: Id,
    _uops: Vec<UOp>,
}

impl UOpEGraph {
    pub fn new(sink: &UOp) -> Self {
        let egraph = EGraph::default();
        let mut ret = UOpEGraph {
            egraph,
            sink: Id::default(),
            _uops: vec![],
        };
        ret.sink = ret.add(sink);
        ret.egraph.rebuild();
        if *GRAPH {
            ret.egraph.dot().to_svg("/tmp/net.egg.svg").unwrap();
            println!("saved egraph to /tmp/net.egg.svg")
        }
        ret
    }

    fn add(&mut self, uop: &UOp) -> Id {
        let src = uop.src.iter().map(|x| self.add(x)).collect();
        let expr = UOpLang {
            op: uop.op.clone(),
            dtype: uop.dtype.clone(),
            src,
            arg: uop.arg.clone(),
        };
        return self.egraph.add(expr);
    }

    fn graph_rewrite(self, pm: &[Rewrite<UOpLang, ()>]) -> RecExpr<UOpLang> {
        let runner = Runner::default().with_egraph(self.egraph).run(pm);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(self.sink);
        best_expr
    }
}

#[cfg(test)]
mod test_tiny {
    use super::*;
    fn sink(src: Vec<UOp>) -> UOp {
        UOp {
            op: UOps::SINK,
            dtype: None,
            src,
            arg: None,
        }
    }

    #[test]
    fn test_uop_eq() {
        let u0: UOp = 0.into();
        let u1: UOp = 0.into();
        assert_eq!(u0, u1)
    }

    #[test]
    fn test_tiny_add() {
        let add = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![42.into(), 1.into()],
            arg: Some((BinaryOps::ADD as u32).to_string()),
        };
        let uegraph = UOpEGraph::new(&add);
        let pat: Pattern<UOpLang> = "(+ ?x 1)".parse().unwrap();
        let matches = pat.search(&uegraph.egraph);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_symbol_lang_ref() {
        let mut egraph: EGraph<SymbolLang, ()> = Default::default();
        let a = egraph.add(SymbolLang::leaf("42"));
        let b = egraph.add(SymbolLang::leaf("1"));
        egraph.add(SymbolLang::new("*", vec![a, b]));
        egraph.rebuild();
        let pat: Pattern<SymbolLang> = "(* ?x 1)".parse().unwrap();
        let matches = pat.search(&egraph);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_tiny_mul() {
        let add = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![42.into(), 1.into()],
            arg: Some((BinaryOps::MUL as u32).to_string()),
        };
        let pm = &[
            rw!("mul-1"; "(* ?x 1)" => "?x"),
            rw!("mul-0"; "(* ?x 0)" => "0"),
        ];
        let uegraph = UOpEGraph::new(&add);
        let sink = uegraph.graph_rewrite(pm);
        panic!("{:?}", sink);
    }

    #[test]
    fn test_symbol_lang_ref_mul() {
        let mut egraph: EGraph<SymbolLang, ()> = Default::default();
        let a = egraph.add(SymbolLang::leaf("42"));
        let b = egraph.add(SymbolLang::leaf("1"));
        let foo = egraph.add(SymbolLang::new("*", vec![a, b]));
        egraph.rebuild();
        let pm = &[
            rw!("mul-1"; "(* ?x 1)" => "?x"),
            rw!("mul-0"; "(* ?x 0)" => "0"),
        ];
        let runner = Runner::default().with_egraph(egraph).run(pm);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(foo);
        panic!("{:?}", best_expr);
    }
}
