from classes.framerate_manager import FramerateManager
from classes.pixel_meter_conversion import Pixel2Meters
from classes.digit_manager import DigitManager

def main():

    parameters_path = "./src/parameters/project.json"
    
    rate_manager = FramerateManager(parameters_path)
    rate_manager.manage_framerate()

    unit_manager = Pixel2Meters(parameters_path)
    unit_manager.apply_conversions()
   
    digit_manager =DigitManager(parameters_path)
    digit_manager.manage_digit_number()
    

if __name__ == "__main__":
    main()
