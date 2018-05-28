import numpy

class Species:
    def __init__(self, *species):
        self._species = Species.create_specie_dict(species)
        self._size = self.size()


    @staticmethod
    def create_specie_dict(species):
        return_dict = {}
        for i, specie in enumerate(species):
            return_dict[specie] = i
        return return_dict
        

    def binary_array_from_specie(self, specie):
        if not specie in self._species:
            return

        byte_array = [0 for _ in self._species]
        byte_array[self._species[specie]] = 1
        return byte_array


    def size(self):
        return len(self._species)
        
    def get_binary_from_list(self, nplist):
        species = []
        for row in nplist:
            for value in row:
                if value in self._species:
                    species.append(self.binary_array_from_specie(value))
        
        return numpy.array(species)


    def get_binary_all_possible(self):
        species = []
        for specie in self._species:
            species.append(self.binary_array_from_specie(specie))

        return numpy.array(species)


    def labels(self):
        return [key for key in self._species.keys()]

