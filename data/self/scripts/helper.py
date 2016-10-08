#coding:utf-8

import os

def load_id_to_name():
    """
    @Des: Load id-name pair
    @ret: id2name : dictionary
    """
    filePath = "name2id.txt"
    id2name = {}
    with open(filePath) as fp:
        for line in fp:
            items = line.strip().split()
            id2name[items[0]] = items[1]
    return id2name

def person_static():
    """
    @Des: Statistics of the genuine & forgery signatures number and print it.
    @Ret: None
    """
    FilePath = "all.txt"
    IDs = {}
    with open(FilePath) as fp:
        for line in fp:
            if line.startswith("SignatureID"):
                ids = line.strip().split()[1]
                personId = ids.split("_")[0]
                sigId = ids.split("_")[1]
                lable = ids.split("_")[2]

                if personId not in IDs:
                    IDs[personId] = [0, 0]
                index = 0 if lable == "1" else 1
                IDs[personId][index] += 1

    genuineCount = 0
    forgeryCount = 0

    for (key, value) in IDs.items():
        print "%s \t: %s" % (id2name[key], value)
        genuineCount += int(value[0])
        forgeryCount += int(value[1])

    print genuineCount, forgeryCount

def split_all_to_files():
    """
    @Des: Split all.txt to files which contain only one signature per file. \
            All genuine signatures will be put in \"genuine\" folders and \
            all forgery signatures will be put in \"forgery\" folders.
            File name format is writerID_sigID.
            Original file contain all signatures.
          
    """
    class Signature:
        def __init__(self, lable, personId):
            self.lable = lable
            self.personId = personId
            self._id = None
            self.lines = None

        def writeToFile(self):
            if self.lines is None:
                raise Exception("Lines could not be None")
            folder = "genuines" if self.lable == "1" else "forgeries"
            folder = "../" + folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = os.path.join(folder, self._id)
            with open(path, "w") as fp:
                for line in self.lines:
                    fp.write(line)

    # Original file path.
    filePath = "../raw_data/SignaturePadRecords-qing.txt"
    personToSigs = {}

    with open(filePath) as fp:
        lines = []
        signature = None
        for line in fp:
            if line.startswith("Signature"):
                if signature is not None:
                    signature.lines = lines
                    lines = []
                ids = line.strip().split()[1]
                personId = ids.split("_")[0]
                sigId = ids.split("_")[1]
                lable = ids.split("_")[2]
                name = id2name[personId]

                signature = Signature(lable, "%s" % '{:03}'.format(int(personId)))
                if name not in personToSigs:
                    personToSigs[name] = [[],[]]
                index = 0 if lable == "1" else 1
                personToSigs[name][index].append(signature)
                continue
            lines.append(line)
        if signature is not None:
            signature.lines = lines

    for sigs in personToSigs.values():
        genuines = sigs[0]
        forgeries = sigs[1]
        for i in range(len(genuines)):
            genuines[i]._id = "%s_%s" % (genuines[i].personId, '{:003}'.format(i))
            genuines[i].writeToFile()
        for i in range(len(forgeries)):
            forgeries[i]._id = "%s_%s" % (forgeries[i].personId, '{:003}'.format(i))
            forgeries[i].writeToFile()

# Globals
id2name = load_id_to_name()

if __name__ == "__main__":
    split_all_to_files()
    # person_static()
