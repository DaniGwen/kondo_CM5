- How to Build? 

	• cd /Kondo-KHR-3HV/build
		○ make clean
		○ make
	• Build with python3 in cython folder(cd /Kondo-KHR-3HV/cython)
		○ python3 setup.py build_ext --inplace
			§ If  it does not build do a clean up: sudo rm -rf build *.so *.c *.cpp

	• Example run:
		○ sudo python3 run_motion.py
