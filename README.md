## Data to download
- [MS COCO pictures](https://cocodataset.org/#download)
- [Localised narratives captions](https://google.github.io/localized-narratives/)
- nltk wordnet: nltk.download('omw-1.4') nltk.download('wordnet')
- stanza: stanza.download('en') 

We use stanza form nltk  for parsing captions. For each noun, gemma-3-4b-it is used to extract the right sysnset in context
 
## Datafiles

- v-coco_HOI.csv
Contains for each annotated image of v-coco the image id, bbox and roles
- coco_train2017_synsets.json, coco_val2017_synsets.json, localized_narratives_train_synsets.json, localized_narratives_val_synsets.json
Contains for each caption the list of nouns (identified using stanza) and the corresponding in context sysnet (using gemma)
- coco_output.csv
Contains object size, depth, ditance to center, colour saliency, HOI, category, deprel and % mention  

## Run code

- wsd_pipeline to create *synsets.json files 
