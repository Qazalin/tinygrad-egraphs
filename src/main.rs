use egg::{rewrite as rw, *};
fn main() {
    // Initialize an empty e-graph
    let mut egraph: EGraph<SymbolLang, ()> = Default::default();
    let a = egraph.add(SymbolLang::leaf("a"));
    let b = egraph.add(SymbolLang::leaf("b"));
    let four = egraph.add(SymbolLang::leaf("4"));
    let add_ab = egraph.add(SymbolLang::new("+", vec![a, b]));
    let add_ab_4 = egraph.add(SymbolLang::new("+", vec![add_ab, four]));
    let foo_node = egraph.add(SymbolLang::new("foo", vec![add_ab_4]));
    egraph.rebuild();

    let pat: Pattern<SymbolLang> = "(+ ?x 4)".parse().unwrap();
    let matches = pat.search(&egraph);
    println!("{:?}", matches);
}
