# HPCProj

### Running

First, boot up the image with Docker:
```bash
# docker build -t hpc-dev . # if not already built
docker run -it --rm -v "$(pwd)":/workspace hpc-dev
```

After that, you can build with *MPI* and *OpenMP*:
```bash
mpicc -fopenmp src/stencil_parallel.c -o prova
```

> Many other flags are missing to do the best optimization

To then run the file, use:
```bash
mpirun --allow-run-as-root ./prova
```

Valid flag options are (values between [] are the default values):  
- `-x`: x size of the plate [10000]
- `-y`: y size of the plate [10000]
- `-e`: how many energy sources on the plate [4]
- `-E`: how much energy per source [1.0]
- `-n`: how many iterations [1000]
- `-p`: whether periodic boundaries applies [0 = false]