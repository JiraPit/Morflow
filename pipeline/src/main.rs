fn main() {
    let code = morflow::cli::run_cli(std::env::args_os());
    std::process::exit(code);
}
