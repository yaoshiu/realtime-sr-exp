{
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      {
        devShell =
          with pkgs;
          mkShell {
            packages = [
              (python3.withPackages (
                ps: with ps; [
                  (opencv4.override {
                    enableGtk3 = true;
                  })
                  numpy
                  tqdm
                  torch
                  setuptools
                  torchvision
                  natsort
                  flatbuffers
                ]
              ))
              ffmpeg
            ];

            shellHook = ''
              export EXTRA_CCFLAGS="-I/usr/include"
              export CUDA_PATH=${pkgs.cudatoolkit}
              export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
            '';
          };
      }
    );
}
