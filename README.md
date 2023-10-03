# cudnnxx

cuDNN C++ wrapper.

## Features

- Header only.
- Descriptors are automatically destroyed via RAII.

## Requirements

- x86_64 Linux. (Tested on Ubuntu 20.04 LTS)
- C++14 compliant compiler (Tested on GCC 9.3)
- CUDA Toolkit 11.0 or later.
- cuDNN 7.6.5 or later.
- GoogleTest 1.10.0 or later.

## Build and test

```
make
```

## Usage

Include cudnnxx/cudnnxx.h.  
See cudnnxx/example_test.cc for more information.

## License

MIT

