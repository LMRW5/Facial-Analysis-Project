from phenotypeSearcher import PhenotypeSearcher


searcher = PhenotypeSearcher()


results = searcher.search(
"tools/processing/stored-faces/face_color.jpg")


for result in results:
    print(result)