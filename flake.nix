{
  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
    in  {
    devShell.x86_64-linux =
      pkgs.mkShell {
        buildInputs = with pkgs; [ 
          opencl-headers
          ocl-icd
          cargo
          cargo-flamegraph
          clang
          fontconfig
          ffmpeg
          gcc
          gdb
          libclang
          pkg-config
          rustc
          rustfmt
          rust-analyzer
        ];
        shellHook = ''
          export LIBCLANG_PATH=${pkgs.libclang.lib}/lib
        '';
      };
    };
}
