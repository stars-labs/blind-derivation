# SPDX-FileCopyrightText: 2021 Serokell <https://serokell.io/>
#
# SPDX-License-Identifier: CC0-1.0
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    { nixpkgs, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          lib,
          ...
        }:
        let
          # Configure nixpkgs to allow unfree packages (required for CUDA)
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };

          # Check if we're on Linux (CUDA only works on Linux)
          isLinux = pkgs.stdenv.isLinux;
        in
        {
          devShells.default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
            RUST_BACKTRACE = 1;

            nativeBuildInputs = with pkgs; [
              pkg-config
              nixfmt-rfc-style
              nixd
              rustc
              cargo
              rust-analyzer
              clippy
              rustfmt
            ];

            buildInputs = with pkgs; [
              openssl
            ] ++ lib.optionals isLinux [
              # CUDA packages (Linux only)
              cudatoolkit
              linuxPackages.nvidia_x11
              libGLU
              libGL
              zlib
              ncurses5
              stdenv.cc
            ];

            shellHook = lib.optionalString isLinux ''
              export CUDA_PATH=${pkgs.cudatoolkit}
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"

              # Add system CUDA driver to library path (for cudarc dynamic loading)
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}

              # For cudarc/rustacuda
              export CUDA_COMPUTE_CAP=86  # Adjust for your GPU (86 = RTX 30xx)

              echo "CUDA environment configured"
              echo "CUDA_PATH=$CUDA_PATH"
              echo "LD_LIBRARY_PATH includes: /run/opengl-driver/lib"
              nvcc --version 2>/dev/null || echo "nvcc not in PATH, but CUDA libs available"
            '';
          };
        };
    };
}
