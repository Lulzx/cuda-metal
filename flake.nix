{
  description = "CuMetal PTX experiment tools; Apple SDK and Metal runtime remain system-provided";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
        buildInputs = [ pkgs.lz4 pkgs.zstd ];
        # Keep the host compiler and SDK from the same Apple installation.
        # This avoids mixing Nix libc++ headers with the selected Apple SDK.
        CC = "/usr/bin/clang";
        CXX = "/usr/bin/clang++";
        OBJCXX = "/usr/bin/clang++";
        shellHook = ''
          export SDKROOT="$(/usr/bin/xcrun --sdk macosx --show-sdk-path)"
          echo "PTX-only shell: configure with -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=TRUE"
          echo "Runtime MSL compilation needs no offline metal/metallib tools."
          echo "For CUDA C++/NVVM import, use the full LLVM setup in docs/build.md."
        '';
      };
    };
}
