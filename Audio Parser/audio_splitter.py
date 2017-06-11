from pydub import AudioSegment

audioFilePrefix = 'AudioFiles/'			# Audio File location
labeledFilePrefix = 'LabeledText/'			# full files
storeDirectory = './SplitAudioFixed_1s/'     # SplitAudio full files

#labeledFilePrefix = 'LabeledText4s/'		# 4 s files
#storeDirectory = './SplitAudio_1s/'     # SplitAudio full files

names = ['M_0017_19y2m_1', 'M_0065_20y1m_1', 'M_0065_20y1m_2', 'M_0065_20y1m_3', 'M_0073_09y11m_1', 
		'M_0073_09y11m_2', 'M_1017_11y8m_1', 'M_0104_18y1m_1', 'M_0078_17y11m_1', 'M_0065_20y7m_1',
		'F_0818_14y2m_1', 'F_0101_17y2m_1', 'F_0101_15y2m_1', 'F_0101_14y8m_1']

for name in names:
	print "**** NEW FILE *****"
	print name
	audioName = audioFilePrefix + name + ".wav"
	textName = labeledFilePrefix + name + ".txt"
	song = AudioSegment.from_wav(audioName)
	open_=open(textName,"r")
	lines=open_.readlines();
	for i in range(0, len(lines)):
		print lines[i]
		if i != 0:
			tokens = lines[i].split()
			if len(tokens) == 0: continue
			label = tokens[0]
			t1 = (float(tokens[1])) * 1000
			t2 = (float(tokens[2])) * 1000

			if t2 - t1 <= 1.0:
				newAudio = song[t1:t2]
				if label is 'U':
					label = 1
				else:
					label = 0
				newName = str(label) + '-' + name + '_' + str(int(round(float(tokens[1])))) + '_' + str(int(round(float(tokens[2])))) + ".wav"
				newAudio.export(storeDirectory + newName, format='wav')
			else:
				# print tokens[1] * 60
				# print tokens[2] * 60
				numIters = ((float(tokens[2])) - (float(tokens[1])))/1
				print str(int(numIters))
				prevNum = float(tokens[1]) * 1000
				for j in range(int(numIters)):
					curr = prevNum + 1000
					newAudio = song[prevNum:curr]
					if label is 'U':
						label = 1
					else:
						label = 0
					newName = str(label) + '-' + name + '_' + str(int(round(float(prevNum)/1000))) + '_' + str(int(round(float(curr)/1000))) + ".wav"
					newAudio.export(storeDirectory + newName, format='wav')
					prevNum = curr

				if label is 'U':
					label = 1
				else:
					label = 0
				newAudio = song[prevNum:t2]
				newName = str(label) + '-' + name + '_' + str(int(round(float(prevNum)/1000))) + '_' + str(int(round(float(tokens[2])))) + ".wav"
				newAudio.export(storeDirectory + newName, format='wav')


