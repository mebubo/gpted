let

nixpkgs = import <nixpkgs> {};
pkgs = nixpkgs.pkgs;

in

pkgs.mkShell {
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/
  '';
}
