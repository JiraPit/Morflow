fn main() {
    let code = pipeline::cli::run_cli(std::env::args_os());
    std::process::exit(code);
}
