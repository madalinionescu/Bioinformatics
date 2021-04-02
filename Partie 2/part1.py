import numpy as np
import itertools as it
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import Image

""" Change pandas options to display the whole matrix. Float number
	formatting prevents displaying anomalities while using V and W
	Matrices which also contain inf values
"""
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.set_option('display.width', 100)
pd.options.display.float_format = '{:,.2f}'.format #no decimal printed

sequence_dict = dict()
matrix_dict = dict()

gap = '-'; match = '|'; mismatch = ' '; positive = ':'; negative = '.'
#Use : for match, . for positive and <space> for negative to match LALIGN format

class Sequence :
	""" Amino acids sequence. Use __str__ to get a string of the a.a.
		sequence and pretty_print to print it in the fasta format.
	"""
	# separe info in the description as list of attributs?
	# if yes which format is used? (Genbank, EMBL, DDBJ)
	def __init__ (self, acids, description = 'Empty sequence') :
		self._description = description.rstrip('\n')
		self._acids = list(acids)
	def __str__ (self) :
		return self._description+'\n'+(''.join(self._acids))+'\n'
	def __getitem__ (self, position) :
		if type(position) == type(1) :
			return self._acids[position]
		else :
			return self._acids.index(position)
	def pretty_print(self) :
		print(self._description+'\n'+(''.join(self._acids)))
	def insert(self, position, character) :
		self._acids.insert(position, character)
	def to_list(self) :
		""" Return the amino acids sequence as list
		"""
		return self._acids
	def len(self) :
		return len(self._acids)

class Matrix :
	""" Enhanced matrix supporting lines/columns labeling, element
		retrieval using this labeling and better string representation.
	"""
	# no need to check the amino acids used to label rows/ columns
	# since they come from a trustworthy source, right?
	def __init__(self, column_acids, row_acids, matrix) :
		self._columns = Sequence(column_acids)
		self._rows = Sequence(row_acids)
		self._matrix = np.matrix(matrix)
	def __str__(self) :
		df = pd.DataFrame(data = self._matrix, index = self._rows, columns = self._columns)
		return df.to_string()
	def __getitem__(self, acids) :
		return self._matrix[self._rows[acids[0]], self._columns[acids[1]]]

class Substitution_Matrix(Matrix) :
	""" Matrix specialization for substitution matrices. 
	"""
	def __init__(self, description, acids, matrix) :
		Matrix.__init__(self, acids, acids, matrix)
		self._description = description
	def pretty_print(self) :
		print(self._description + Matrix.__str__(self))

class Score_Matrix(Matrix) :
	def __init__(self, seq1, seq2, penalty, subst_matrix) :
		""" Initially, the main matrix contains only zeros, while V and W
			contain zeros, with the exception of the first column/ row, which
			is fille with a value equivalent to negative infinity.
			Attention : penalty is positive
		"""
		matrix = [[0 for i in range(seq1.len() + 1)] for j in range(seq2.len() + 1)]
		Matrix.__init__(self, seq1, seq2, matrix)
		self._columns.insert(0,'-')
		self._rows.insert(0,'-')
		self._penalty = penalty		   #a couple : I = penalty[0], E = penalty[1]
		self._subst_matrix = subst_matrix #a substitution matrix
		
		self._V = [[0 for i in range(self._columns.len())] for j in range(self._rows.len())]
		for i in range(self._columns.len()) :
			self._V[0][i] = - math.inf	#first row contains - infinity
			
		self._W = [[0 for i in range(self._columns.len())] for j in range(self._rows.len())]
		for j in range(self._rows.len()) :
			self._W[j][0] = -math.inf	 #first column contains - infinity
			
	def __str__(self) :
		""" Print will display all the three matrices successively.
		"""
		df1 = pd.DataFrame(data = self._matrix, index = self._rows.to_list(), columns = self._columns.to_list())
		if hasattr(self, '_V') and hasattr(self, '_W') :
			df2 = pd.DataFrame(data = self._V, index = self._rows.to_list(), columns = self._columns.to_list())
			df3 = pd.DataFrame(data = self._W, index = self._rows.to_list(), columns = self._columns.to_list())
			return 'Score matrix:\n' + df1.to_string() + '\n\n' + \
				   'V matrix:\n' + df2.to_string() + '\n\n' + \
				   'W matrix:\n' + df3.to_string()
		else :
			return 'Score matrix:\n' + df1.to_string() + '\n'
	
	def get_optimal_result(self) :
		if hasattr(self, '_global_solution') and len(self._global_solution) != 0 :
			return self._global_solution[0]
		elif hasattr(self, '_local_solution') and len(self._local_solution) != 0 :
			return self._local_solution[0]

	def Needleman_Wunsch(self) :
		""" Implementation of the Needleman-Wunsch to calculate the score 
			matrix values using the affine gap penalty.
		"""
		for i in range(1, self._rows.len()) :
			for j in range(1, self._columns.len()) :
				self._V[i][j] = max((self._matrix[i-1, j] - self._penalty[0]), (self._V[i-1][j] - self._penalty[1]))
				self._W[i][j] = max((self._matrix[i, j-1] - self._penalty[0]), (self._W[i][j-1] - self._penalty[1]))
				self._matrix[i, j] = max((self._matrix[i-1, j-1] + self._subst_matrix[self._rows[i], self._columns[j]])\
								  , self._V[i][j], self._W[i][j])

	def extremity_gaps(self) :
		""" Calculates the gapes scores on the first line/ column of 
			the score matrix.
		"""
		self._matrix[0, 1] =  - self._penalty[0]
		self._matrix[1, 0] = - self._penalty[0]
		for i in range(2, self._columns.len()) :
			self._matrix[0, i] = self._matrix[0, i-1] - self._penalty[1]
		for i in range(2, self._rows.len()) :
			self._matrix[i, 0] = self._matrix[i-1, 0] - self._penalty[1]

	def align_globally(self, k, i, j, seq1 = '', seq2 = '', aligner = '', matches = 0, similar = 0) :
		""" Rebuilding the path backwards to obtain the global alignment.
			To be called initially with the maximum values for i and j (lower-right corner)
		"""
		if i == 0 and j == 0 :
			if len(self._global_solution) < k :
				matches, similar, score = self.compute_score(seq1, matches, similar)
				self._global_solution.append([seq1,aligner,seq2,matches,similar,score])
		elif i == 0 and j > 0 : #on the upper line -> computed from the left value is the only possibility
			self.align_globally(k, i, j-1, self._columns[j]+seq1, gap+seq2, mismatch+aligner, matches, similar)
		elif i > 0 and j == 0 : #on the left edge -> computed from the top value is the only possibility
			self.align_globally(k, i-1, j, gap+seq1, self._rows[i]+seq2, mismatch+aligner, matches, similar)
		else :				  #no gap introduced
			if self._matrix[i-1, j-1] + self._subst_matrix[self._rows[i], self._columns[j]] == self._matrix[i, j] :
				if self._columns[j] == self._rows[i] :
					self.align_globally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, match+aligner, matches+1, similar)
				else :
					if self._subst_matrix[self._rows[i], self._columns[j]] >= 0 :
						self.align_globally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, positive+aligner, matches, similar+1)
					else :
						self.align_globally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, negative+aligner, matches, similar)
			if self._V[i][j] == self._matrix[i, j] : #introducing gap in the second sequence (columns sequence)
				self.align_globally(k, i-1, j, gap+seq1, self._rows[i]+seq2, mismatch+aligner, matches, similar)
			if self._W[i][j] == self._matrix[i, j] : #introducing gap in the first sequence (rows sequence)
				self.align_globally(k, i, j-1, self._columns[j]+seq1, gap+seq2, mismatch+aligner, matches, similar)

	def global_alignment(self, k = 3) :
		""" Call the functions in the right order to obtain the global alignment.
		"""
		self._global_solution = list()
		self.extremity_gaps()
		self.Needleman_Wunsch()
		i = self._rows.len() - 1
		j = self._columns.len() - 1
		self.align_globally(k, i, j)

	def semiglobal_alignment(self, k = 3) :
		self._global_solution = list()
		self.Needleman_Wunsch()
		i = self._rows.len() - 1
		j = self._columns.len() - 1
		self.align_globally(k, i, j)

	def pretty_print(self) :
		""" Prints the aligned sequences in similar format to the one used
			by LALIGN : | = match
						: = positive score
						. = negative score
			The matching characters can be changed to match exactly the LALIGN
			output
		"""
		if hasattr(self, '_global_solution') :
			for i in self._global_solution :
				for j in range(0,len(i[0]),80) :
					print(i[0][j:j+80] + '\n' + i[1][j:j+80] + '\n' + i[2][j:j+80])
				print('NW score: ', i[5])
				print('Identity score: {:,.1f}%'.format(i[3]))
				print('Similarity score: {:,.1f}%'.format(i[4]))
		elif hasattr(self, '_local_solution') :
			for i in self._local_solution :
				for j in range(0,len(i[0]),80) :
					print(i[0][j:j+80] + '\n' + i[1][j:j+80] + '\n' + i[2][j:j+80])
				print(i[0] + '\n' + i[1] + '\n' + i[2])
				print('SW score: ', i[5])
				print('Identity score: {:,.1f}%'.format(i[3]))
				print('Similarity score: {:,.1f}%'.format(i[4]))
		elif hasattr(self, '_Profile_solution') :
			for i in self._Profile_solution :
				position = '{} - {}: '.format(i[1],i[2])
				print(position, end='')
				for j in range(0,len(i[0]),80) :
					if j == 0 :
						print(i[0][j:j+80])
					else :
						print(' '*len(position)+i[0][j:j+80])

	def Smith_Waterman(self, k = 1, l = 1, recompute = 0) :
		if hasattr(self, '_last') :
			k = self._last[0]
			l = self._last[1]
			recompute = 1
		for i in range(k, self._rows.len()) :
			for j in range(l, self._columns.len()) :
				if (recompute == 1 and self._matrix[i,j] != 0) != 0 or recompute == 0 :
					self._V[i][j] = max((self._matrix[i-1, j] - self._penalty[0]), (self._V[i-1][j] - self._penalty[1]))
					self._W[i][j] = max((self._matrix[i, j-1] - self._penalty[0]), (self._W[i][j-1] - self._penalty[1]))
					self._matrix[i, j] = max((self._matrix[i-1, j-1] + self._subst_matrix[self._rows[i], self._columns[j]])\
									  , self._V[i][j], self._W[i][j], 0)

	def get_max(self) :
		""" Returns the maximum score in the matrix and its position.
		"""
		u = 0
		v = 0
		maximum = -math.inf
		for i in range(self._rows.len()) :
			for j in range(self._columns.len()) :
				if self._matrix[i, j] >= maximum :
					maximum = self._matrix[i, j]
					u = i
					v = j
		return maximum, u, v

	def compute_score(self, seq, matches, similar, global_score = True) :
		""" Calculates the identity score and the similarity score when a 
			complete path is found.
		"""
		similar = (matches + similar) / len(seq) * 100
		matches = matches / len(seq) * 100
		if global_score :
			score = self._matrix[-1, -1]
			return matches, similar, score
		else :
			return matches, similar

	def replace_with_zero(self, i, j) :
		self._matrix[i, j] = 0
		if hasattr(self, '_V') and hasattr(self, '_W') :
			self._V[i][j] = 0
			self._W[i][j] = 0

	def align_locally(self, k, i, j, seq1 = '', seq2 = '', aligner = '', matches = 0, similar = 0, score = 0, parent = [-1,-1]) :
		if self._matrix[i, j] == 0 :
			if len(seq1) >= 2 and len(self._local_solution) < k :
				matches, similar = self.compute_score(seq1, matches, similar, global_score = False)
				self._local_solution.append([seq1,aligner,seq2,matches,similar,score])
				self._last = parent
		else :
			if self._matrix[i-1, j-1] + self._subst_matrix[self._rows[i], self._columns[j]] == self._matrix[i, j] :
				if self._columns[j] == self._rows[i] :
					self.align_locally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, match+aligner, matches+1, similar, score, [i,j])
				else :
					if self._subst_matrix[self._rows[i], self._columns[j]] >= 0 :
						self.align_locally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, positive+aligner, matches, similar+1, score, [i,j])
					else :
						self.align_locally(k, i-1, j-1, self._columns[j]+seq1, self._rows[i]+seq2, negative+aligner, matches, similar, score, [i,j])
			if self._V[i][j] == self._matrix[i, j] :
				self.align_locally(k, i-1, j, self._rows[i]+seq1, gap+seq2, mismatch+aligner, matches, similar, score, [i,j])
			if self._W[i][j] == self._matrix[i, j] :
				self.align_locally(k ,i, j-1, gap+seq1, self._columns[j]+seq2, mismatch+aligner, matches, similar, score, [i,j])
			self.replace_with_zero(i,j)

	def local_alignment(self, k = 3) :
		""" Repeats the local alignment algorithm until the demanded number of 
			alignments are found or until there are no possible local alignments
			in the matrix.
		"""
		self._local_solution = list()
		self.Smith_Waterman()
		self._last = [1,1]
		maximum, i, j = self.get_max()
		while maximum != 0 and len(self._local_solution) < k :
			self.align_locally(k, i, j, score = maximum)
			self.Smith_Waterman()
			maximum, i, j = self.get_max()

def parse_sequence(path, file_name) :
	""" Process the fasta files and store a list Sequence ADT instances
		in the sequence_dict.
	"""
	if path == '.' :
		path = ''
	group = file_name.split('.')
	group = group[0]
	#group denotes the category of the sequence e.g. BRD-sequences
	sequence_dict[group] = list()
	#one entry in the dictionary for each file
	with open(path + file_name, 'r') as fasta_file :
		#better exception handling and automatic closing using with
		first = 1;
		to_build = 0;
		for line in fasta_file :
			if line[0] == '>' or line[0] == ';' : #description
				if first != 1 :
					if to_build == 1 :			#new sequence, old one complete
						#we arrived at the beginning of a new sequence, so we have
						#to instantiate the old one
						sequence_dict[group].append(Sequence(acids, description))
						description = line
						acids = ''
					else :						#continue reading the description
						description += line;
				else :							#a.a. sequence 
					first = 0
					description = line
					acids = ''
			else :
				to_build = 1;
				#to instantiate a sequence we need at least a line representing
				#an amino acids sequence
				acids += line.rstrip('\n')
	if to_build == 1 :
		#instantiate the last sequence from the file
		sequence_dict[group].append(Sequence(acids, description))

def parse_matrix(path, file_name) :
	""" Process the substitution matrix files and a list of Substitution_Matrix
		ADT instances in the matrix dictionary
	"""
	if path == '.' :
		path = ''
	with open(path + file_name, 'r') as matrix_file :
		#with for automatic closing
		description = ''
		matlines = ''
		first = 1
		for line in matrix_file :
			if line[0] == '#' :
				description += line  #build the description
			else :
				if first == 1 :	  #get the amino acids sequence
					column_acids = list(line.replace(' ','').rstrip())
					first = 0
				else :
					line = line[2:]  #ignore the first element (the row label)
					line = line.strip()
					matlines += (line + ';')
					#process the matrix in the format accepted by numpy.matrix :
					#elements separated by space, lines separated by semicolon
		matlines = matlines.strip()
		matlines = matlines.rstrip(';')
		#last lign should not be ending with semicolon or uneven lenegth error will be generated
		matname = file_name.split('.')
		matname = matname[0]
		matrix_dict[matname] = Substitution_Matrix(description, column_acids, matlines)
        
def get_files(path = '.', reprocess = False) :
	""" Process the files in the given path, or in the project folder.
	"""
	files = os.listdir(path) #returns a list with all the file names
							 #in the location given by the path
	for file in files :
		filename = (file.split('.'))
		#filename[0] contains literally the name while filename[0] the extension
		if (filename[0] not in sequence_dict.keys() and filename[0] not in matrix_dict.keys()) or reprocess :
			if filename[1] == 'fasta' :
				parse_sequence(path + '/', file)
			elif filename[1] == 'txt' :
				parse_matrix(path + '/', file)
			#other file types like .ipynb for this notebook are ignored

def test() :
	def test_examples() :
		""" This test tries to align the protein sequences seen in lectures.
			Result table  : ┌───┬─────────┬─────────────────────┬───────────────┐
			(test.fasta	    | # | penalty | substitution matrix | alignment	    |
			contains the	├───┼─────────┼─────────────────────┼───────────────┤
			sequences in	| 1 | I=8 E=8 | blosum62			| THISLINE-	    |
			these tests in  | g |		  |					    | ISALIGNED	    |
			the following   ├───┼─────────┼─────────────────────┼───────────────┤
			order :		    | 2 | I=4 E=4 | blosum62			| THIS-LI-NE-   |
			MGGETFA		    | g |		  |					    | --ISALIGNED   |
			GGVTTF		    ├───┼─────────┼─────────────────────┼───────────────┤
			THISLINE		| 3 | I=12 E=2| blosum62			| -GGVTTF	    |
			ISALIGNED	    | g |		  |					    | MGGETFA	    | 
							├───┼─────────┼─────────────────────┼──────────┬────┤
							| 3 | I=4 E=4 | blosum62			| IS-LI-NE | IN |
							| l |		  |					    | ISALIGNE | IS |
							└───┴─────────┴─────────────────────┴──────────┴────┘

		"""
		ok = 1
		if test(sequence_dict['test'][2], sequence_dict['test'][3], [8,8],\
				matrix_dict['blosum62'], ['THISLINE-','ISALIGNED']) == 0 :
			ok = 0
		if test(sequence_dict['test'][2], sequence_dict['test'][3], [4,4],\
				matrix_dict['blosum62'], ['THIS-LI-NE-','--ISALIGNED']) == 0 :
			ok = 0
		if test(sequence_dict['test'][0], sequence_dict['test'][1], [12,2],\
				matrix_dict['blosum62'], ['-GGVTTF','MGGETFA']) == 0 :
			ok = 0
		if test(sequence_dict['test'][2], sequence_dict['test'][3], [4,4],\
				matrix_dict['blosum62'], ['IS-LI-NE','ISALIGNE','IN','IS'],\
				glob = 0) == 0 :
			ok = 0
		if ok == 1 :
			print("Examples tests ok!\n")

	def test_lalign() :
		ok = 1
		if test(sequence_dict['BRD-sequence'][1], sequence_dict['BRD-sequence'][4], [4,1],\
				matrix_dict['blosum62'],\
				['K-HAAYAWPFYKPVD-VEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVV',\
				 'KSHQS-AWPFMEPVKRTEAPG---YYEVIRFPMDLKTMSERLKNRYYVSKKLFMADLQRVFTNCKEYNPPESEYY']) == 0 :
			ok = 0
		if test(sequence_dict['BRD-sequence'][0],sequence_dict['BRD-sequence'][5], [12,2],\
				matrix_dict['blosum62'],\
				['WKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIV',\
				'SGRRLCDLFM--VKPSKKDYPDYYKIILEPMDLKIIEHNIRNDKYAGEEGMIEDMKLMFRNARHYNEEGSQVY']) == 0 :
			ok = 0
		if test(sequence_dict['BRD-sequence'][2], sequence_dict['BRD-sequence'][3], [8,2],\
				matrix_dict['blosum62'],\
				['RDLPNTYPFHTPVNAKVVKDYYKIITRPMDLQTLRENVRKRLYPSREEFREHLELIVKNSATYNGPKHSLT',
				 'MAVPDSWPFHHPVNKKFVPDYYKVIVNPMDLETIRKNISKHKYQSRESFLDDVNLILANSVKYNGPESQYT']) == 0 :
			ok = 0
		if test(sequence_dict['protein-sequences'][0], sequence_dict['BRD-sequence'][6], [12,2],\
				matrix_dict['blosum62'], \
				['QRKDPHGFFAFPVTDAIAPGYSMIIKHPMDFGTMKDKIVANEYKSVTEFKADFKLMCDNAMTYNRPDTVYY',\
				 'QRKDPSAFFSFPVTDFIAPGYSMIIKHPMDFSTMKEKIKNNDYQSIEELKDNFKLMCTNAMIYNKPETIYY',\
				 'HPVDLSSLSSKL', 'HPMDFSTMKEKI'],\
				glob = 0) == 0 :
			ok = 0
		if ok == 1 :
			print("LALIGN tests ok!")

	def test(seq1, seq2, penalty, mat, sol, glob = 1) :
		test = Score_Matrix(seq1, seq2, penalty, mat)
		if glob == 1 :
			test.global_alignment(3)
			ok = False
			for tested_sol in test._global_solution :
				ok = ok or not((tested_sol[0] != sol[0] and tested_sol[0] != sol[1])\
				or (tested_sol[2] != sol[0] and tested_sol[2] != sol[1]))
			if not ok :
				print('─────────────────── GLOBAL ALIGNMENT TEST FAILED ! ───────────────────')
				test.pretty_print()
				print('──────────────────────────────────────────────────────────────────────\n')
				return 0
			else :
				print('────────────────── GLOBAL ALIGNMENT TEST SUCCESSFUL ──────────────────')
				test.pretty_print()
				print('──────────────────────────────────────────────────────────────────────\n')
				return 1
		else :
			test.local_alignment(2)
			if (test._local_solution[0][0] != sol[0] and test._local_solution[0][0] != sol[1])\
			or (test._local_solution[0][2] != sol[0] and test._local_solution[0][2] != sol[1])\
			or (test._local_solution[1][0] != sol[2] and test._local_solution[1][0] != sol[3])\
			or (test._local_solution[1][2] != sol[2] and test._local_solution[1][2] != sol[3]) :
				print('──────────────────── LOCAL ALIGNMENT TEST FAILED ! ───────────────────')
				test.pretty_print()
				print('──────────────────────────────────────────────────────────────────────\n')
				return 0
			else :
				print('────────────────── LOCAL ALIGNMENT TEST SUCCESSFUL ───────────────────')
				test.pretty_print()
				print('──────────────────────────────────────────────────────────────────────\n')
				return 1

	test_examples()
	test_lalign()

get_files()