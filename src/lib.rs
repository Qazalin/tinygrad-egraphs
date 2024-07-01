use egg::{rewrite as rw, *};
use half::{bf16, f16};
use num_enum::{FromPrimitive, IntoPrimitive};
use serde::{Deserialize, Serialize};
use serde_pickle as pickle;
use std::os::raw::{c_char, c_int, c_uchar};
use std::{any::Any, convert::Infallible, env, fmt::Display, slice};

lazy_static::lazy_static! {
    pub static ref GRAPH: bool = env::var("GRAPH").map(|v| v == "1").unwrap_or(false);
    pub static ref DEBUG: bool = env::var("DEBUG").map(|v| v == "1").unwrap_or(false);
}

// *** tinygrad
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

fn parse_value(dtype: &str, val_str: &str) -> Result<Box<dyn Any>, String> {
    fn parse<T: 'static + std::str::FromStr>(val_str: &str) -> Result<Box<dyn Any>, String>
    where
        T::Err: ToString,
    {
        val_str
            .parse::<T>()
            .map(|v| Box::new(v) as Box<dyn Any>)
            .map_err(|e| e.to_string())
    }

    match dtype {
        "bool" => parse::<bool>(val_str),
        "char" => parse::<i8>(val_str),
        "unsigned_char" => parse::<u8>(val_str),
        "short" => parse::<i16>(val_str),
        "unsigned_short" => parse::<u16>(val_str),
        "int" => parse::<i32>(val_str),
        "unsigned_int" => parse::<u32>(val_str),
        "long" => parse::<i64>(val_str),
        "unsigned_long" => parse::<u64>(val_str),
        "half" => parse::<f16>(val_str),
        "__bf16" => parse::<bf16>(val_str),
        "float" => parse::<f32>(val_str),
        "double" => parse::<f64>(val_str),
        _ => panic!("{dtype}"),
    }
}
static INTEGERS: &[&str] = &[
    "char",
    "unsigned_char",
    "short",
    "unsigned_short",
    "int",
    "unsigned_int",
    "long",
    "unsigned_long",
];
static FLOATS: &[&str] = &["half", "__bf16", "float", "double"];

// *** egraph
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
        self.op == other.op
            && self.len() == other.len()
            && self.arg == other.arg
            && self.dtype == other.dtype
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
        // https://github.com/bytecodealliance/wasm-tools/blob/1cf71f9a9bbb4706e1e5d21ad2a12586a3911199/crates/wasm-mutate/src/mutators/peephole/rules.rs#L52
        // TODO: bool alus shouldn't have dtype
        let splits: Vec<&str> = op.split(".").collect();
        let dtype = splits[0];
        if let Some(_) = parse_value(dtype, splits[1]).ok() {
            return Ok(Self {
                op: UOps::CONST,
                dtype: Some(format!("dtypes.{dtype}")),
                src: vec![],
                arg: Some(splits[1].to_string()),
            });
        }
        let operator = match splits[1] {
            "+" => BinaryOps::ADD as u32,
            "*" => BinaryOps::MUL as u32,
            "//" => BinaryOps::IDIV as u32,
            "-" => UnaryOps::NEG as u32,
            "max" => BinaryOps::MUL as u32,
            _ => todo!("{op}"),
        };
        Ok(Self {
            op: UOps::ALU,
            dtype: Some(format!("dtypes.{dtype}")),
            src: children,
            arg: Some(operator.to_string()),
        })
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

    fn uops(self, pm: &[Rewrite<UOpLang, ()>]) -> Vec<UOp> {
        let expr = self.graph_rewrite(pm);
        let mut uops: Vec<UOp> = vec![];
        expr.as_ref().iter().for_each(|ul| {
            let src: Vec<&UOp> = ul.src.iter().map(|x| &uops[usize::from(*x)]).collect();
            uops.push(UOp {
                op: ul.op.clone(),
                dtype: ul.dtype.clone(),
                src: src.iter().map(|x| (**x).clone()).collect(),
                arg: ul.arg.clone(),
            })
        });
        // assert_eq!(uops.last().unwrap().op, UOps::SINK, "didn't end with sink");
        uops
    }
}

macro_rules! rwd {
    (
        $dt:expr,
        $name:expr;
        $lhs:tt => $rhs:tt
        $(if $cond:expr)*
    ) => {{
        let lhs = $lhs.replace("{dt}", $dt).parse::<Pattern<_>>().unwrap();
        let rhs = $rhs.replace("{dt}", $dt).parse::<Pattern<_>>().unwrap();
        rw!($name.replace("{dt}", $dt); lhs => rhs)
    }};
}
fn pattern_matcher() -> Vec<Rewrite<UOpLang, ()>> {
    let mut rules = vec![];
    INTEGERS.iter().chain(FLOATS.iter()).for_each(|dt| {
        rules.extend([
            // Communative properties https://github.com/jafioti/luminal/blob/8d36e703d70082cddd9a627bef7533036c60ab25/src/shape/symbolic.rs#L903C1-L909C62
            rwd!(dt, "{dt}.commute-add"; "({dt}.+ ?a ?b)" => "({dt}.+ ?b ?a)"),
            rwd!(dt, "{dt}.commute-mul"; "({dt}.* ?a ?b)" => "({dt}.* ?b ?a)"),
            rwd!(dt, "{dt}.commute-max"; "({dt}.max ?a ?b)" => "({dt}.max ?b ?a)"),
            // ** self folding **
            rwd!(dt, "{dt}.add-0"; "({dt}.+ ?x {dt}.0)" => "?x"),
            // TODO: x - 0
            rwd!(dt, "{dt}.mul-0"; "({dt}.* ?x {dt}.0)" => "{dt}.0"),
            rwd!(dt, "{dt}.mul-1"; "({dt}.* ?x {dt}.1)" => "?x"),
        ]);
        rules.extend(match INTEGERS.contains(dt) {
            true => vec![
                rwd!(dt, "{dt}.idiv-self"; "({dt}.// ?x ?x)" => "?x"),
                rwd!(dt, "{dt}.idiv-one"; "({dt}.// ?x {dt}.1)" => "?x"),
            ],
            // TODO: (UOp.var('x') / UOp.cvar('c'), lambda x,c: x*exec_alu(UnaryOps.RECIP, c.dtype, [c.arg])),    # x/c -> x*(1/c)
            false => vec![rwd!(dt, "{dt}.recip-self"; "({dt}.// ?x ?x)" => "?x")],
        });
        if !dt.starts_with("unsigned") {
            rules.push(rwd!(dt, "{dt}.mul-neg"; "({dt}.* ?x {dt}.-1)" => "({dt}.- ?x)"));
            if INTEGERS.contains(dt) {
                rules.push(rwd!(dt, "{dt}.idiv-neg"; "({dt}.// ?x {dt}.-1)" => "({dt}.- ?x)"));
            }
        }
    });
    rules.push(rw!("bool.max"; "(bool.max ?x bool.false)" => "?x"));
    rules
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
            if *DEBUG {
                println!("{:?}", uop);
            }
            let uegraph = UOpEGraph::new(&uop);
            let uops = uegraph.uops(&pattern_matcher());
            let ret = serde_pickle::to_vec(&uops, serde_pickle::SerOptions::new()).unwrap();
            let len = ret.len();
            let ptr = ret.as_ptr() as *mut c_char;
            std::mem::forget(ret);
            ByteArray { ptr, len }
        }
        Err(err) => {
            let value =
                pickle::from_slice::<serde_pickle::Value>(bytes, serde_pickle::DeOptions::new())
                    .unwrap();
            panic!("couldn't parse {:?} as uop\n error = {:?}", value, err);
        }
    }
}

#[cfg(test)]
mod test_tiny {
    use super::*;
    fn sink(src: &[UOp]) -> UOp {
        UOp {
            op: UOps::SINK,
            dtype: None,
            src: src.to_vec(),
            arg: None,
        }
    }

    impl From<i32> for UOp {
        fn from(value: i32) -> Self {
            UOp {
                op: UOps::CONST,
                dtype: Some("dtypes.int".into()),
                src: vec![],
                arg: Some(value.to_string()),
            }
        }
    }
    impl From<f32> for UOp {
        fn from(value: f32) -> Self {
            UOp {
                op: UOps::CONST,
                dtype: Some("dtypes.float".into()),
                src: vec![],
                arg: Some(value.to_string()),
            }
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
        let pat: Pattern<UOpLang> = "(int.+ ?x int.1)".parse().unwrap();
        let matches = pat.search(&uegraph.egraph);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_tiny_mul_float() {
        let s = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.float".into()),
            src: vec![(42.0).into(), (0.0).into()],
            arg: Some((BinaryOps::MUL as u32).to_string()),
        };
        let uegraph = UOpEGraph::new(&s);
        let uops = uegraph.uops(&pattern_matcher());
        assert_eq!(uops.len(), 1);
        assert_eq!(uops[0], (0.0).into());
    }

    #[test]
    fn test_tiny_mul() {
        let mul_1 = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![42.into(), 1.into()],
            arg: Some((BinaryOps::MUL as u32).to_string()),
        };
        let mul_0 = UOp {
            op: UOps::ALU,
            dtype: Some("dtypes.int".into()),
            src: vec![69.into(), 0.into()],
            arg: Some((BinaryOps::MUL as u32).to_string()),
        };
        let pm = &[
            rw!("mul-1"; "(int.* ?x int.1)" => "?x"),
            rw!("mul-0"; "(int.* ?x int.0)" => "int.0"),
        ];
        let uegraph = UOpEGraph::new(&sink(&[mul_1, mul_0]));
        let uops = uegraph.uops(pm);
        assert_eq!(uops.len(), 3);
        let sinked = &uops.last().unwrap().src;
        assert_eq!(sinked.len(), 2);
        assert_eq!(sinked[0], 42.into());
        assert_eq!(sinked[1], 0.into());
    }

    #[test]
    fn test_symbol_lang_ref_mul() {
        type EGraph = egg::EGraph<SymbolLang, ()>;
        #[derive(Debug)]
        struct ConstFolderApplier {
            x: &'static str,
            y: &'static str,
        }
        impl Applier<SymbolLang, ()> for ConstFolderApplier {
            fn apply_one(
                &self,
                egraph: &mut EGraph,
                _: Id,
                subst: &Subst,
                _: Option<&PatternAst<SymbolLang>>,
                _: Symbol,
            ) -> Vec<Id> {
                let x_nodes = &egraph[subst[self.x.parse().unwrap()]].nodes;
                let y_nodes = &egraph[subst[self.y.parse().unwrap()]].nodes;
                assert_eq!(x_nodes.len(), 1);
                assert_eq!(y_nodes.len(), 1);
                let ret = x_nodes[0].op.as_str().parse::<i32>().unwrap()
                    * y_nodes[0].op.as_str().parse::<i32>().unwrap();
                let new_id = egraph.add(SymbolLang::leaf(ret.to_string()));
                return vec![new_id];
            }
        }
        let mut egraph: EGraph = Default::default();
        let a = egraph.add(SymbolLang::leaf("42"));
        let b = egraph.add(SymbolLang::leaf("2"));
        let foo = egraph.add(SymbolLang::new("*", vec![a, b]));
        egraph.rebuild();
        let pm = &[
            rw!("mul-1"; "(* ?x 1)" => "?x"),
            rw!("mul-const"; "(* ?x ?y)" => { ConstFolderApplier { x: "?x", y: "?y" } }),
        ];
        let runner = Runner::default().with_egraph(egraph).run(pm);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(foo);
        panic!("{:?}", best_expr);
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
}
