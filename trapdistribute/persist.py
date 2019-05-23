
import typing as t

import datetime
import os

from mtgorp.db.database import CardDatabase
from mtgorp.models.serilization.serializeable import Serializeable, serialization_model, Inflator
from mtgorp.models.serilization.strategies.jsonid import JsonId
from mtgorp.utilities.containers import HashableMultiset

from magiccube.laps.traps.trap import Trap

from misccube import paths


class TrapCollection(Serializeable):

	def __init__(self, traps: t.Iterable[Trap]):
		self._traps = HashableMultiset(traps)

	@property
	def traps(self) -> HashableMultiset[Trap]:
		return self._traps

	@property
	def minimal_string_list(self) -> str:
		return '\n'.join(
			sorted(
				trap.node.minimal_string
				for trap in
				self._traps
			)
		)

	def serialize(self) -> serialization_model:
		return {
			'traps': self._traps,
		}

	@classmethod
	def deserialize(cls, value: serialization_model, inflator: Inflator) -> 'Serializeable':
		return cls(
			(
				Trap.deserialize(trap, inflator)
				for trap in
				value.get('traps', ())
			)
		)

	def __hash__(self) -> int:
		return hash(self._traps)

	def __eq__(self, other: object) -> bool:
		return (
			isinstance(other, self.__class__)
			and self._traps == other._traps
		)


class TrapCollectionPersistor(object):

	_OUT_DIR = os.path.join(paths.OUT_DIR, 'trap_collections')
	TIMESTAMP_FORMAT = '%y_%m_%d_%H_%M_%S'

	def __init__(self, db: CardDatabase):
		self._db = db
		self._strategy = JsonId(self._db)

	def get_all_trap_collections(self) -> t.Iterator[TrapCollection]:
		if not os.path.exists(self._OUT_DIR):
			os.makedirs(self._OUT_DIR)

		trap_collections = os.listdir(self._OUT_DIR)

		if not trap_collections:
			return

		names_times = [] #type: t.List[t.Tuple[str, datetime.datetime]]

		for name in trap_collections:
			try:
				names_times.append(
					(
						name,
						datetime.datetime.strptime(
							os.path.splitext(name)[0],
							self.TIMESTAMP_FORMAT,
						),
					)
				)
			except ValueError:
				pass

		if not names_times:
			return

		sorted_pairs = sorted(names_times, key=lambda item: item[1], reverse = True)

		for name, time in sorted_pairs:
			with open(os.path.join(self._OUT_DIR, name), 'r') as f:
				yield self._strategy.deserialize(TrapCollection, f.read())

	def get_most_recent_trap_collection(self) -> t.Optional[TrapCollection]:
		all_collections = self.get_all_trap_collections()
		try:
			return all_collections.__next__()
		except StopIteration:
			return None

	def get_trap_collection(self, name: str) -> TrapCollection:
		with open(os.path.join(self._OUT_DIR, name), 'r') as f:
			return self._strategy.deserialize(TrapCollection, f.read())

	def persist(self, trap_collection: TrapCollection):
		if not os.path.exists(self._OUT_DIR):
			os.makedirs(self._OUT_DIR)

		with open(
			os.path.join(
				self._OUT_DIR,
				datetime.datetime.strftime(
					datetime.datetime.today(),
					self.TIMESTAMP_FORMAT,
				),
			) + '.json',
			'w',
		) as f:

			f.write(
				JsonId.serialize(
					trap_collection
				)
			)
