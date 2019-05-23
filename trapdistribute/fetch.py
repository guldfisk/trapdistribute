import typing as t


from sheetclient.client import GoogleSheetClient

from mtgorp.db.database import CardDatabase

from magiccube.laps.traps.tree.parse import PrintingTreeParser, PrintingTreeParserException

from misccube.trapification.algorithm import ConstrainedNode
from misccube import values



class ConstrainedCubeablesFetchException(Exception):
	pass



class ConstrainedNodeFetcher(object):
	SHEET_NAME = 'trapables'

	_value_value_map = {
		0: 0,
		1:  1,
		2: 5,
		3: 15,
		4: 30,
		5: 55,
	}

	_legal_groups = {
		'WHITE',
		'BLUE',
		'BLACK',
		'RED',
		'GREEN',
		'drawgo',
		'mud',
		'post',
		'midrange',
		'mill',
		'reanimate',
		'burn',
		'hatebear',
		'removal',
		'lock',
		'yardvalue',
		'ld',
		'storm',
		'tezz',
		'lands',
		'shatter',
		'bounce',
		'shadow',
		'stifle',
		'beat',
		'cheat',
		'pox',
		'counter',
		'discard',
		'cantrip',
		'balance',
		'stasis',
		'standstill',
		'whitehate',
		'bluehate',
		'blackhate',
		'redhate',
		'greenhate',
		'antiwaste',
		'delirium',
		'sacvalue',
		'lowtoughnesshate',
		'armageddon',
		'stax',
		'bloom',
		'weldingjar',
		'drawhate',
		'pluscard',
		'ramp',
		'devoteddruid',
		'fetchhate',
		'dragon',
		'company',
		'naturalorder',
		'flash',
		'wincon',
		'vial',
		'fixing',
		'colorlessvalue',
		'fetchable',
		'indestructable',
		'legendarymatters',
		'sol',
		'manland',
		'storage',
		'croprotate',
		'dnt',
		'equipment',
		'livingdeath',
		'eggskci',
		'hightide',
		'fatty',
		'walker',
		'blink',
		'miracles',
		'city',
		'wrath',
		'landtax',
		'discardvalue',
		'edict',
		'phoenix',
		'enchantress',
		'dork',
		'tinker',
		'landtax',
		'highpowerhate',
		'affinity',
		'academy',
		'stompy',
		'shardless',
		'lanterns',
		'depths',
		'survival',
		'landstill',
		'moat',
		'combo',
		'kite',
		'haste',
		'fog',
	}

	def __init__(self, db: CardDatabase, document_id: str = values.DOCUMENT_ID):
		self._db = db

		self._printing_tree_parser = PrintingTreeParser(db)
		self._sheet_client = GoogleSheetClient(document_id)

	def _parse_groups(self, s: str) -> t.FrozenSet[str]:
		groups = []

		if s:
			for group in s.split(','):
				group = group.rstrip().lstrip().replace('-', '')

				if not group in self._legal_groups:
					raise ConstrainedCubeablesFetchException(f'"{group}" not a legal group')

				groups.append(group)

		return frozenset(groups)

	def _fetch(self, start_column: int) -> t.List[ConstrainedNode]:
		exceptions = []
		constrained_cubeables = []

		for row in self._sheet_client.read_sheet(
			self.SHEET_NAME,
			start_column = start_column,
			start_row = 4,
			end_column = start_column + 3,
			end_row = 1000,
		):
			if not row:
				print('Empty row in trapables input!')
				# TODO should probably be a runtime warning, but that requires logging...
				continue

			amount_cell, printings_cell, value_cell = row[:3]
			groups_cell = row[3] if len(row) > 3 else ''

			try:
				amount = int(amount_cell)
			except ValueError:
				amount = 1

			try:
				node = self._printing_tree_parser.parse(printings_cell)
			except PrintingTreeParserException as e:
				exceptions.append(e)
				continue

			try:
				value = self._value_value_map[int(value_cell)]
			except (ValueError, KeyError) as e:
				exceptions.append(e)
				continue

			try:
				groups = self._parse_groups(groups_cell)
			except ConstrainedCubeablesFetchException as e:
				exceptions.append(e)
				continue

			for _ in range(amount):
				constrained_cubeables.append(
					ConstrainedNode(
						node = node,
						value = value,
						groups = groups,
					)
				)

		if exceptions:
			raise ConstrainedCubeablesFetchException(exceptions)

		return constrained_cubeables

	def fetch_garbage(self) -> t.List[ConstrainedNode]:
		return self._fetch(1)

	def fetch_garbage_lands(self) -> t.List[ConstrainedNode]:
		return self._fetch(5)

	def fetch_all(self) -> t.List[ConstrainedNode]:
		garbage = self.fetch_garbage()
		garbage.extend(self.fetch_garbage_lands())
		return garbage