from pip_module_scanner.scanner import Scanner, ScannerException

try:
   scanner = Scanner(path='C:\\Users\\Nobody\\Desktop\\BD\\data-augmentation', output="requirements.txt")
   scanner.run()

   for lib in scanner.libraries_found:
       print ("Found module %s at version %s" % (lib.key, lib.version))

except ScannerException as e:
    print("Error: %s" % str(e))