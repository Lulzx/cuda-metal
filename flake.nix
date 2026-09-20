{
  description = "CuMetal development tools for Apple Silicon";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      pkgs = import nixpkgs { system = "aarch64-darwin"; };
      llvm = pkgs.llvmPackages_21;
      sdk = pkgs.apple-sdk_15;
      cudaClang = pkgs.writeShellScript "cumetal-cuda-clang++" ''
        # The NVPTX pass does not inherit Darwin's implicit C header search.
        # Keep libc++ and Clang's resource headers ahead of the SDK headers.
        exec ${llvm.clang}/bin/clang++ \
          -idirafter "${sdk}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include" "$@"
      '';
      appleTool = name: pkgs.writeShellScriptBin name ''
        # Apple's host linker needs its own SDK, not Nix's split SDK/libc++.
        exec /usr/bin/env -u SDKROOT /usr/bin/${name} "$@"
      '';
      appleTools = pkgs.symlinkJoin {
        name = "cumetal-apple-tools";
        # Test scripts use BSD mktemp's `-t prefix` form.
        paths = map appleTool [ "xcrun" "xcodebuild" "mktemp" ];
      };
    in
    {
      devShells.aarch64-darwin.default =
        (pkgs.mkShell.override { stdenv = llvm.stdenv; }) {
          # Clang's NVPTX target rejects this host-only compiler option.
          hardeningDisable = [ "zerocallusedregs" ];
          packages = with pkgs; [
            cmake
            ninja
            pkg-config
            bashInteractive
            python3
            llvm.llvm
          ];
          buildInputs = [
            llvm.llvm.dev
            sdk
            pkgs.libffi
            pkgs.libxml2
            pkgs.lz4
            pkgs.zstd
          ];

          LLVM_DIR = "${llvm.llvm.dev}/lib/cmake/llvm";
          CMAKE_PREFIX_PATH = pkgs.lib.concatStringsSep ":" [
            "${llvm.llvm.dev}"
            "${pkgs.lib.getDev pkgs.libffi}"
            "${pkgs.lib.getLib pkgs.libffi}"
            "${pkgs.lib.getDev pkgs.libxml2}"
            "${pkgs.lib.getLib pkgs.libxml2}"
            "${pkgs.lib.getDev pkgs.lz4}"
            "${pkgs.lib.getLib pkgs.lz4}"
            "${pkgs.lib.getDev pkgs.zstd}"
            "${pkgs.lib.getLib pkgs.zstd}"
          ];
          # CUDA sources also include C++ headers, so retain Nix's SDK wrapper.
          # The compiler and older test scripts consult different variable names.
          CUMETAL_CUDA_CLANG = cudaClang;
          CUMETAL_CLANG = cudaClang;
          CUMETAL_CUDA_CLANG_21 = cudaClang;

          shellHook = ''
            # Direct compiler calls in tests need the same sysroot as CMake.
            export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -isysroot $SDKROOT"
            # Keep Nix's SDKROOT for host compilation, but use the selected Apple
            # installation for Metal instead of Nix's xcbuild/xcrun replacement.
            export PATH="${appleTools}/bin:$PATH"
            export DEVELOPER_DIR="$(/usr/bin/env -u DEVELOPER_DIR /usr/bin/xcode-select -p)"
            if ! /usr/bin/xcrun --find metal >/dev/null 2>&1 ||
               ! /usr/bin/xcrun --find metallib >/dev/null 2>&1; then
              echo "Apple Metal command-line tools are unavailable; install/select Xcode and its Metal Toolchain for AOT tests." >&2
            fi
          '';
        };
    };
}
