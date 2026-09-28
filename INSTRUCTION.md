Design Document: Dynamic Data Pipeline (Phase 1)

1. Project Overview & Objectives

The goal of this project is to build a high-performance, modular data processing engine in Rust. The system will process heavy data payloads (large vectors, images, audio) using a node-based architecture where individual processing steps ("Actions") can be swapped in and out at runtime without recompiling the main application.
Phase 1 Scope (MVP)

To establish a stable foundation, Phase 1 is strictly limited to proving the FFI (Foreign Function Interface) boundary.

    Host: Runs a single, hardcoded action.

    Action: An identity block that takes a payload and returns it unmodified.

    Data Flow: Guarantee zero-copy memory transfer between the host and the dynamically loaded library.

2. Architecture & Technical Decisions

The project is structured as a Cargo Workspace with three distinct domains:

    core_types (Library): The central source of truth. Defines the data structures and function signatures used to cross the FFI boundary.

    actions/* (Dynamic Libraries): Precompiled actions (.so/.dll). They depend only on core_types.

    pipeline (Executable): The host engine. It dynamically loads actions into memory and pushes data through them.

Technical Stack

    Dynamic Linking (libloading): Used by the pipeline to load .so/.dll files at runtime.

    ABI Stability (abi_stable): Rust does not have a stable ABI. To prevent segmentation faults when passing complex types (like Vectors) across the FFI boundary, all boundary data structures will use abi_stable types (e.g., RVec instead of Vec).

    Memory Strategy: Zero-copy. Payloads are passed by transferring ownership of the RVec pointer across the boundary.

3. Directory Structure
   Plaintext

my_project/
├── Cargo.toml  
├── core_types/  
│ ├── Cargo.toml
│ └── src/lib.rs
├── pipeline/  
│ ├── Cargo.toml
│ └── src/main.rs
└── actions/  
 └── identity/  
 ├── Cargo.toml
└── src/lib.rs

Workspace Cargo.toml
Ini, TOML

[workspace]
members = [
"core_types",
"pipeline",
"actions/identity",
]
resolver = "2"

4. Implementation Guide (Phase 1)
   Step 1: The Core Vocabulary (core_types)

This crate defines what data exists in the pipeline. For V1, we will just use a generic RawBytes type.

core_types/Cargo.toml
Ini, TOML

[package]
name = "core_types"
version = "0.1.0"
edition = "2021"

[dependencies]
abi_stable = "0.11"

core_types/src/lib.rs
Rust

use abi_stable::std_types::{RString, RVec};
use abi_stable::StableAbi;

#[repr(u8)] #[derive(StableAbi, Debug, PartialEq, Eq)]
pub enum DataType {
RawBytes,
}

#[repr(C)] #[derive(StableAbi, Debug)]
pub enum Payload {
Data {
buffer: RVec<u8>,
},
Error(RString),
}

// Universal function signatures for all Actions
pub type ProcessFn = extern "C" fn(Payload) -> Payload;
pub type GetTypeFn = extern "C" fn() -> DataType;

Step 2: The Action (actions/identity)

This action must be compiled as a C-dynamic library. It simply accepts a payload and returns it.

actions/identity/Cargo.toml
Ini, TOML

[package]
name = "identity"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"] # Compiles to .so / .dll

[dependencies]
core_types = { path = "../../core_types" }

actions/identity/src/lib.rs
Rust

use core_types::{DataType, Payload};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
DataType::RawBytes
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
DataType::RawBytes
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
// Identity action: do absolutely nothing and return the payload.
// This proves we can safely pass memory back and forth!
payload
}

Step 3: The Host Engine (pipeline)

The runtime environment that loads the identity block and triggers it.

pipeline/Cargo.toml
Ini, TOML

[package]
name = "pipeline"
version = "0.1.0"
edition = "2021"

[dependencies]
core_types = { path = "../core_types" }
libloading = "0.8"
abi_stable = "0.11"

pipeline/src/main.rs
Rust

use std::env::consts::{DLL_PREFIX, DLL_EXTENSION};
use libloading::{Library, Symbol};
use core_types::{Payload, ProcessFn, GetTypeFn};
use abi_stable::std_types::RVec;

fn get_action_path(action_name: &str) -> String {
format!("../target/debug/{}{}.{}", DLL_PREFIX, action_name, DLL_EXTENSION)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
let action_name = "identity";
let path = get_action_path(action_name);

    println!("Starting Pipeline Engine...");
    println!("Loading action from: {}", path);

    unsafe {
        // 1. Load the dynamic library
        let lib = Library::new(&path)?;

        // 2. Extract function pointers
        let get_in: Symbol<GetTypeFn> = lib.get(b"get_input_type")?;
        let get_out: Symbol<GetTypeFn> = lib.get(b"get_output_type")?;
        let process: Symbol<ProcessFn> = lib.get(b"process")?;

        println!("Loaded [{}] | Input: {:?} -> Output: {:?}", action_name, get_in(), get_out());

        // 3. Create a dummy payload
        let initial_data = Payload::Data {
            buffer: RVec::from(vec![10, 20, 30, 40]),
        };

        println!("Sending payload: {:?}", initial_data);

        // 4. Execute the action
        let result = process(initial_data);

        println!("Received payload: {:?}", result);
    }

    Ok(())

}

5. Future Roadmap (Phase 2 & Beyond)

Once Phase 1 successfully compiles and runs without memory errors, the project will expand to:

    Pipeline Parser: Introduce a string/JSON parser to dynamically build an array of actions (e.g., ["load_audio", "identity", "export"]).

    Action Chaining: Update the pipeline to hold multiple libloading::Library instances in a struct and loop through their process functions sequentially.

    Runtime Type Checking: Implement validation in the host to ensure get_output_type() of Block A matches get_input_type() of Block B before the pipeline begins executing.
