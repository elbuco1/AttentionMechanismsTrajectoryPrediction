from classes.dataset_extractor import SddExtractor


def main():
    
    raw_extractor = SddExtractor("./src/parameters/project.json")
    raw_extractor.extract()

    

    
    
    

if __name__ == "__main__":
    main()