#[allow(unused)]
use egg::{rewrite as rw, *};
use num_enum::{FromPrimitive, IntoPrimitive};
use serde::{Deserialize, Serialize};
use serde_pickle as pickle;
use std::os::raw::{c_char, c_int, c_uchar};
use std::slice;

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
#[derive(Serialize, Deserialize, Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
struct UOp {
    op: UOps,
    dtype: Option<String>,
    src: Vec<UOp>,
    arg: Option<String>,
}
impl UOp {
    fn const_<T: ToString>(dtype: T, x: u32) -> UOp {
        return UOp {
            op: UOps::CONST,
            dtype: Some(dtype.to_string()),
            src: vec![],
            arg: Some(x.to_string()),
        };
    }
    fn alu<T: Into<u32>>(op: T, src: Vec<UOp>) -> UOp {
        let op: u32 = op.into();
        return UOp {
            op: UOps::ALU,
            dtype: Some(src.last().unwrap().dtype.clone().unwrap()),
            src,
            arg: Some(op.to_string()),
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
        Ok(uop) => {
            println!("{:?}", uop);
            let new = UOp::const_("dtypes.int".to_string(), 42);
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
    egraph: EGraph<SymbolLang, ()>,
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
        ret.sink = ret.add(&sink);
        ret.egraph.rebuild();
        return ret;
    }

    fn add(&mut self, uop: &UOp) -> Id {
        let src = uop.src.iter().map(|x| self.add(x)).collect();
        let expr = match uop.op {
            UOps::SINK => SymbolLang::new("sink", src),
            UOps::VAR => panic!("didn't expect var here"),
            UOps::DEFINE_GLOBAL => todo!("{uop:?}"),
            UOps::DEFINE_VAR => todo!("{uop:?}"),
            UOps::DEFINE_LOCAL => todo!("{uop:?}"),
            UOps::DEFINE_ACC => todo!("{uop:?}"),
            UOps::CONST => SymbolLang::leaf(uop.arg.clone().unwrap()),
            UOps::SPECIAL => SymbolLang::leaf("special"),
            UOps::NOOP => todo!("{uop:?}"),
            UOps::UNMUL => todo!("{uop:?}"),
            UOps::GEP => todo!("{uop:?}"),
            UOps::CAST => todo!("{uop:?}"),
            UOps::BITCAST => todo!("{uop:?}"),
            UOps::ALU => {
                let op = uop.arg.clone().unwrap().parse::<u32>().unwrap();
                let op = match uop.src.len() {
                    1 => match UnaryOps::from(op) {
                        UnaryOps::EXP2 => "exp2",
                        UnaryOps::LOG2 => "log2",
                        UnaryOps::CAST => "cast",
                        UnaryOps::BITCAST => "bitcast",
                        UnaryOps::SIN => "sin",
                        UnaryOps::SQRT => "sqrt",
                        UnaryOps::NEG => "-",
                        UnaryOps::RECIP => "recip",
                    },
                    2 => match BinaryOps::from(op) {
                        BinaryOps::ADD => "+",
                        BinaryOps::MUL => "*",
                        BinaryOps::IDIV => "//",
                        BinaryOps::MAX => "max",
                        BinaryOps::MOD => "%",
                        BinaryOps::CMPLT => "<",
                        BinaryOps::CMPNE => "!=",
                        BinaryOps::XOR => "^",
                        BinaryOps::SHR => ">>",
                        BinaryOps::SHL => "<<",
                    },
                    3 => match TernaryOps::from(op) {
                        TernaryOps::WHERE => "where",
                        TernaryOps::MULACC => "mulacc",
                    },
                    _ => panic!(),
                };
                SymbolLang::new(op, src)
            }
            UOps::WMMA => todo!("{uop:?}"),
            UOps::LOAD => todo!("{uop:?}"),
            UOps::STORE => todo!("{uop:?}"),
            UOps::PHI => todo!("{uop:?}"),
            UOps::BARRIER => todo!("{uop:?}"),
            UOps::IF => todo!("{uop:?}"),
            UOps::RANGE => todo!("{uop:?}"),
            UOps::ENDRANGE => todo!("{uop:?}"),
            UOps::ENDIF => todo!("{uop:?}"),
        };
        return self.egraph.add(expr);
    }

    fn graph_rewrite(self, pm: &[Rewrite<SymbolLang, ()>]) -> Vec<UOp> {
        let runner = Runner::default().with_egraph(self.egraph).run(pm);
        // Extract the optimized expression
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(self.sink);
        let mut uops: Vec<UOp> = vec![];
        best_expr.as_ref().iter().for_each(|x| {
            let src = x
                .children
                .iter()
                .map(|x| return uops[x.to_string().parse::<usize>().unwrap()].clone())
                .collect::<Vec<UOp>>();
            let uop = match x.op.as_str() {
                "+" => UOp::alu(BinaryOps::ADD, src),
                "max" => UOp::alu(BinaryOps::MAX, src),
                "sink" => UOp {
                    op: UOps::SINK,
                    dtype: None,
                    src: vec![],
                    arg: None,
                },
                "special" => UOp {
                    op: UOps::SPECIAL,
                    dtype: Some("dtyps.int".into()),
                    src: vec![],
                    arg: Some("".into()),
                },
                _ => UOp::const_("dtypes.int", x.op.to_string().parse().unwrap()),
            };
            uops.push(uop)
        });
        assert_eq!(uops.last().unwrap().op, UOps::SINK, "didn't end with sink");
        return uops;
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
        let u0 = UOp::const_("dtypes.int", 0);
        let u1 = UOp::const_("dtypes.int", 0);
        assert_eq!(u0, u1)
    }

    #[test]
    fn test_tiny_add() {
        let val = UOp::const_("dtypes.int", 42);
        let pat: Pattern<SymbolLang> = "(+ ?x 1)".parse().unwrap();
        let add = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![val.clone(), UOp::const_("dtypes.int", 1)],
            arg: Some((BinaryOps::ADD as u32).to_string()),
        };
        let uegraph = UOpEGraph::new(&add);
        let matches = pat.search(&uegraph.egraph);
        assert!(!matches.is_empty());

        let add = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![val.clone(), UOp::const_("dtypes.into", 2)],
            arg: Some((BinaryOps::ADD as u32).to_string()),
        };
        let uegraph = UOpEGraph::new(&add);
        let matches = pat.search(&uegraph.egraph);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_more_serious_pm() {
        let pm = &[
            rw!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
            rw!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),
            rw!("add-0"; "(+ ?x 0)" => "?x"),
            rw!("mul-0"; "(* ?x 0)" => "0"),
            rw!("mul-1"; "(* ?x 1)" => "?x"),
        ];
        let s0 = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![UOp::const_("dtypes.int", 42), UOp::const_("dtypes.int", 1)],
            arg: Some((BinaryOps::MUL as u32).to_string()),
        };
        let s1 = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![s0.clone(), UOp::const_("dtypes.int", 1)],
            arg: Some((BinaryOps::ADD as u32).to_string()),
        };
        let s2 = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![s0.clone(), UOp::const_("dtypes.int", 1)],
            arg: Some((BinaryOps::ADD as u32).to_string()),
        };
        let sink = UOp {
            op: UOps::SINK,
            dtype: None,
            src: vec![s0.clone(), s1, s2],
            arg: None,
        };
        let uegraph = UOpEGraph::new(&sink);
        let uops = uegraph.graph_rewrite(pm);
        assert_eq!(uops.len(), 4);
        assert_eq!(uops[0], UOp::const_("dtypes.int", 42));
        assert_eq!(uops[1], UOp::const_("dtypes.int", 1));
        assert_eq!(uops[2].arg, Some((BinaryOps::ADD as u32).to_string()));
    }

    #[test]
    fn test_serious_pm() {
        let pm: &[Rewrite<SymbolLang, ()>] = &[
            rw!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
            rw!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),
            rw!("add-0"; "(+ ?x 0)" => "?x"),
            rw!("mul-0"; "(* ?x 0)" => "0"),
            rw!("mul-1"; "(* ?x 1)" => "?x"),
            // max on special can go away (TODO: special should be variable, same thing applies)
            // (UOp.max(UOp.cvar('c'), UOp(UOps.SPECIAL).name('s')), lambda c,s: c if (s.arg[2]-1) <= c.arg else None),
            rw!("max-special"; "(max ?x special)" => "special"),
        ];

        let lidx0 = UOp {
            op: UOps::SPECIAL,
            dtype: Some("dtypes.int".into()),
            src: vec![],
            arg: Some("(0, 'lidx0', 3)".into()),
        };

        let alu = UOp::alu(BinaryOps::MAX, vec![lidx0, UOp::const_("dtypes.int", 1)]);
        let uegraph = UOpEGraph::new(&sink(vec![alu]));
        let uops = uegraph.graph_rewrite(pm);
        panic!("{:?}", uops);
    }
}
