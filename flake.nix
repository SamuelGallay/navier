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
          clang
          gcc
          libclang
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
