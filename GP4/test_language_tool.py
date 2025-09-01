import language_tool_python
tool = language_tool_python.LanguageTool("fr-FR")
good_text = "Un texte bon."
bad_text = "Moi hier jsui aller a la magazinne pour acheter des pomplmouss mais yavais plu alors jlai dis a la caishièr que c t vraiment injusteuh. Apré j’ai pries le bus 72 direction centre ville mais le chaufeur roulé bizarre et tt les passager criyé 'arreteeeeee'. Ensuite on a vue un chien qui courai vite-vite-vite derriere une voiture rougeatrement. C’etais troooop chelou, genre incroyablemen bizarre. Après j’ai dormis dans le canapey paske j’avais tropp fatiguéss."
print(good_text)
print(tool.check(good_text))
print(bad_text)
print(tool.check(bad_text))