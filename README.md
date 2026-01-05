# micropyML
micropyML is a micropython Neural Network inference library that is fully C based for fast, low-latency numerical operations 
and fixed-point arithmetic on microcontrollers. 

Docs: https://jay-damodaran.github.io/micropyML/

## Build Process
There may be alternative directory structures that work, but this is what I did to build the firmware. 

1. Clone micropython, ulab, and micropyML into a build directory  
project/  
├── micropython  
├── ulab  
├── micropyML  

2. Copy micropython.cmake out of the buildFiles folder in micropyML and paste at the same level as all of the other directories.  
project/  
├── micropython  
├── ulab  
├── micropyML  
├── micropython.cmake  

3. Follow the micropython build process for your board with setting USER_C_MODULES = .../micropython.cmake in the final build step
4. Flash the resulting firmware file to your board

