import typing as t

import math
import random
import itertools
import statistics
import functools
import operator
import copy

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from mtgorp.models.persistent.attributes.colors import Color
from mtgorp.utilities.containers import HashableMultiset

from magiccube.laps.traps.tree.printingtree import AllNode, PrintingNode
from magiccube.laps.traps.trap import Trap


class UnweightedFitness(base.Fitness):

	weights = ()

	def getValues(self):
		return self.wvalues

	def setValues(self, values):
		self.wvalues = values

	values = property(
		getValues,
		setValues,
		base.Fitness.delValues,
	)


class ConstrainedNode(object):

	def __init__(self, value: float, node: PrintingNode, groups: t.Iterable[str] = ()):
		self.value = value
		self.node = node

		self.groups = frozenset(
			itertools.chain(
				groups,
				(
					color.name
					for color in
					Color
					if (
						len(node.children) == 1
						and node.children.__iter__().__next__().cardboard.front_card.color == frozenset((color,))
					)
				)
			),
		)

		self._hash = hash(self.node)

	def __repr__(self):
		return f'CC({self.node})'

	def __deepcopy__(self, memodict: t.Dict):
		return self


class Individual(object):

	def __init__(self):
		self.fitness = UnweightedFitness()

	@property
	@abstractmethod
	def as_trap_collection(self) -> HashableMultiset[Trap]:
		pass


class TrapDistribution(Individual):

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode] = (),
		trap_amount: int = 1,
		traps: t.Optional[t.List[t.List[ConstrainedNode]]] = None,
		random_initialization: bool = False
	):
		super().__init__()

		self._trap_amount = trap_amount

		if traps is None:
			self.traps = [[] for _ in range(trap_amount)] #type: t.List[t.List[ConstrainedNode]]

			if random_initialization:
				for constrained_node in constrained_nodes:
					random.choice(self.traps).append(constrained_node)

			else:
				for constrained_node, i in zip(constrained_nodes, itertools.cycle(range(trap_amount))):
					self.traps[i].append(constrained_node)

		else:
			self.traps = traps #type: t.List[t.List[ConstrainedNode]]
			self._trap_amount = len(self.traps)

	@property
	def trap_amount(self):
		return self._trap_amount

	@property
	def as_trap_collection(self) -> HashableMultiset[Trap]:
		traps = []

		for trap in self.traps:
			cubeables = []

			if not trap:
				raise Exception('Empty trap')

			for constrained_node in trap:
				cubeable = constrained_node.node
				if isinstance(cubeable, AllNode) and len(cubeable.children) == 1:
					cubeables.extend(cubeable.children)
				else:
					cubeables.append(cubeable)

			traps.append(Trap(AllNode(cubeables)))

		return HashableMultiset(traps)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({self.traps})'


class DistributionDelta(Individual):

	def __init__(
		self,
		origin: TrapDistribution,
		added_nodes: t.Collection[ConstrainedNode],
		removed_node_indexes: t.FrozenSet[int],
		max_trap_difference: int,
		trap_amount_delta: int = 0,
	):
		super().__init__()

		self._origin = origin
		self._added_nodes = added_nodes
		self._removed_node_indexes = removed_node_indexes
		self._max_trap_difference = max_trap_difference

		self.node_moves = {} #type: t.Dict[t.Tuple[int, int], int]
		self.added_node_indexes = {} #type: t.Dict[ConstrainedNode, int]

		self._trap_amount_delta = trap_amount_delta

		self._all_index_set = frozenset(range(self.trap_amount))

		self.removed_trap_redistributions = {} #type: t.Dict[int, t.List[int]]

		for _ in range(self.removed_trap_amount):
			removed_index = self.get_available_trap_index()
			self.removed_trap_redistributions[removed_index] = []

		for removed_index in self.removed_trap_redistributions:
			for _ in self._origin.traps[removed_index]:
				self.removed_trap_redistributions[removed_index].append(
					self.get_available_trap_index()
				)

		for index in itertools.chain(*self.removed_trap_redistributions.values()):
			if index in self.removed_trap_redistributions:
				raise Exception('dude what')

		for node in self._added_nodes:
			self.added_node_indexes[node] = random.sample(
				(
					self.valid_trap_indexes
					if len(self.modified_trap_indexes) < self._max_trap_difference else
					self.modified_trap_indexes - self.removed_trap_redistributions.keys()
				),
				1
			)[0]

	@property
	def origin(self) -> TrapDistribution:
		return self._origin

	@property
	def trap_amount(self) -> int:
		return self._origin.trap_amount + self._trap_amount_delta

	@property
	def added_trap_amount(self) -> int:
		return max(self._trap_amount_delta, 0)

	@property
	def removed_trap_amount(self) -> int:
		return max(-self._trap_amount_delta, 0)

	@property
	def max_trap_difference(self) -> int:
		return self._max_trap_difference

	@property
	def removed_node_indexes(self) -> t.FrozenSet[int]:
		return self._removed_node_indexes

	@property
	def valid_trap_indexes(self) -> t.FrozenSet[int]:
		return self._all_index_set - self.removed_trap_redistributions.keys()

	def get_available_trap_index(self) -> int:
		modifications = self.modified_trap_indexes
		return random.sample(
			(
				self.valid_trap_indexes
				if len(modifications) < self.max_trap_difference else
				modifications - self.removed_trap_redistributions.keys()
			),
			1
		)[0]

	def evacuate_removed_trap(self, index: int) -> None:
		for from_indexes, to_index in self.node_moves.items():
			if to_index == index:
				self.node_moves[from_indexes] = self.get_available_trap_index()

		changes = [] #type: t.List[t.Tuple[int, int]

		for indexes in self.node_moves:
			if indexes[0] == index:
				changes.append(indexes)

		for indexes in changes:
			del self.node_moves[indexes]

		for node, added_node_index in self.added_node_indexes.items():
			if index == added_node_index:
				self.added_node_indexes[node] = self.get_available_trap_index()

		for removed_index, target_indexes in self.removed_trap_redistributions.items():
			if not removed_index == index:
				continue

			for target_index_index, target_index in enumerate(target_indexes):
				if target_index in self.removed_trap_redistributions:
					target_indexes[target_index_index] = self.get_available_trap_index()

	@property
	def modified_trap_indexes(self) -> t.FrozenSet[int]:
		return frozenset(
			itertools.chain(
				self._removed_node_indexes,
				(index for index, _ in self.node_moves),
				self.node_moves.values(),
				self.added_node_indexes.values(),
				range(self.origin.trap_amount, self.origin.trap_amount + self._trap_amount_delta),
				*self.removed_trap_redistributions.values(),
			)
		) - self.removed_trap_redistributions.keys()

	@property
	def trap_distribution(self) -> TrapDistribution:
		modified_distribution = copy.deepcopy(self._origin)

		for _ in range(self.added_trap_amount):
			modified_distribution.traps.append([])

		moves = [] #type: t.List[t.Tuple[ConstrainedNode, int]]

		for index, trap in enumerate(modified_distribution.traps):
			for from_indexes, to_index in sorted(
				(item for item in self.node_moves.items() if item[0][0] == index),
				key = lambda vs: vs[0][1],
				reverse = True,
			):
				moves.append(
					(
						trap.pop(from_indexes[1]),
						to_index,
					)
				)

		for node, index in moves:
			modified_distribution.traps[index].append(node)

		for node, index in self.added_node_indexes.items():
			modified_distribution.traps[index].append(node)

		for from_index, target_indexes in self.removed_trap_redistributions.items():
			for target_index in target_indexes:
				modified_distribution.traps[target_index].append(
					modified_distribution.traps[from_index].pop(0)
				)

		for index in sorted(self.removed_trap_redistributions, reverse=True):
			del modified_distribution.traps[index]

		return modified_distribution

	@property
	def as_trap_collection(self) -> HashableMultiset[Trap]:
		return self.trap_distribution.as_trap_collection


def mutate_trap_distribution(distribution: TrapDistribution) -> t.Tuple[TrapDistribution]:
	for i in range(3):
		selected_group = random.choice(
			[
				group
				for group in
				distribution.traps
				if group
			]
		)
		target_group = random.choice(
			[
				group
				for group in
				distribution.traps
				if group != selected_group
			]
		)
		target_group.append(
			selected_group.pop(
				random.randint(
					0,
					len(selected_group) - 1,
				)
			)
		)
	return distribution,


def mate_distributions(
	distribution_1: TrapDistribution,
	distribution_2: TrapDistribution,
	distributor: 'Distributor',
) -> t.Tuple[TrapDistribution, TrapDistribution]:

	locations = {
		node: []
		for node in
		distributor.constrained_nodes
	}
	for grouping in (distribution_1, distribution_2):
		for i in range(len(grouping.traps)):
			for node in grouping.traps[i]:
				locations[node].append(i)

	for grouping in (distribution_1, distribution_2):
		groups = [[] for _ in range(distributor.trap_amount)]

		for node, possibilities in locations.items():
			groups[random.choice(possibilities)].append(node)

		grouping.traps = groups

	return distribution_1, distribution_2


def mutate_distribution_delta(delta: DistributionDelta) -> t.Tuple[DistributionDelta]:
	for i in range(5):

		if random.random() < .8:
			modified_indexes = delta.modified_trap_indexes

			from_trap_index = random.sample(
				(
					delta.valid_trap_indexes
					if len(modified_indexes) < delta.max_trap_difference else
					modified_indexes - delta.removed_trap_redistributions.keys()
				),
				1
			)[0]

			node_index_options = frozenset(
				range(
					len(
						delta.origin.traps[from_trap_index]
					)
				) if from_trap_index < len(delta.origin.traps) else
				()
			) - frozenset(
				_node_index
				for _trap_index, _node_index in
				delta.node_moves
				if _trap_index == from_trap_index
			)

			if not node_index_options:
				continue

			node_index = random.sample(node_index_options, 1)[0]

			from_trap_index_set = frozenset((from_trap_index,))

			possible_trap_indexes_to = (
				modified_indexes - delta.removed_trap_redistributions.keys() - from_trap_index_set
				if len(modified_indexes | from_trap_index_set) >= delta.max_trap_difference else
				delta.valid_trap_indexes - from_trap_index_set
			)

			if not possible_trap_indexes_to:
				continue

			delta.node_moves[(from_trap_index, node_index)] = random.sample(possible_trap_indexes_to, 1)[0]

		else:

			if not delta.node_moves:
				continue

			del delta.node_moves[random.choice(list(delta.node_moves))]

	if delta.added_node_indexes:
		for i in range(2):

			if random.random() < .2:
				moved_node = random.choice(list(delta.added_node_indexes))
				from_trap_index = delta.added_node_indexes[moved_node]

				del delta.added_node_indexes[moved_node]

				modified_indexes = delta.modified_trap_indexes

				from_trap_index_set = frozenset((from_trap_index,))

				possible_trap_indexes_to = (
					modified_indexes - delta.removed_trap_redistributions.keys() - from_trap_index_set
					if len(modified_indexes) >= delta.max_trap_difference else
					delta.valid_trap_indexes - from_trap_index_set
				)

				if not possible_trap_indexes_to:
					continue

				delta.added_node_indexes[moved_node] = random.sample(possible_trap_indexes_to, 1)[0]

	if delta.removed_trap_redistributions:
		for _ in range(2):
			if random.random() < .05:
				unremoved_index = random.choice(list(delta.removed_trap_redistributions))

				new_removed_index = random.sample(delta.valid_trap_indexes, 1)[0]

				del delta.removed_trap_redistributions[unremoved_index]
				delta.removed_trap_redistributions[new_removed_index] = []

				delta.removed_trap_redistributions[new_removed_index] = [
					delta.get_available_trap_index()
					for _ in delta.origin.traps[new_removed_index]
				]

				for index in delta.removed_trap_redistributions:
					delta.evacuate_removed_trap(index)

		for _ in range(5):
			if random.random() < .1:
				trap = delta.removed_trap_redistributions[
					random.choice(list(delta.removed_trap_redistributions))
				]
				trap[random.randint(0, len(trap) - 1)] = delta.get_available_trap_index()

	return delta,


def mate_distribution_deltas(
	delta_1: DistributionDelta,
	delta_2: DistributionDelta,
) -> t.Tuple[DistributionDelta, DistributionDelta]:

	moves = copy.copy(delta_1.node_moves)
	moves.update(delta_2.node_moves)

	adds = {
		node:
			frozenset(
				(
					delta_1.added_node_indexes[node],
					delta_2.added_node_indexes[node],
				)
			)
		for node in
		delta_1.added_node_indexes
	}

	move_amounts = (len(delta_1.node_moves), len(delta_2.node_moves))
	min_moves = min(move_amounts)
	max_moves = max(move_amounts)

	removed_distributions = {} #type: t.Dict[int, t.List[t.List[int]]]

	deltas = (delta_1, delta_2)

	for delta in deltas:
		for index, indexes in delta.removed_trap_redistributions.items():
			try:
				removed_distributions[index].append(indexes)
			except KeyError:
				removed_distributions[index] = [indexes]


	for delta in deltas:

		modified = set()
		modified.update(delta.removed_node_indexes)

		delta.removed_trap_redistributions = {}
		for index, distribution_options in random.sample(removed_distributions.items(), delta.removed_trap_amount):
			delta.removed_trap_redistributions[index] = []
			modified.add(index)

			for options in zip(*distribution_options):
				if len(modified) < delta.max_trap_difference:
					target_index = random.choice(options)
					delta.removed_trap_redistributions[index].append(target_index)
					modified.add(target_index)

				else:
					moved = False
					options = list(options)
					random.shuffle(options)
					for target_index in options:
						if target_index in modified:
							delta.removed_trap_redistributions[index].append(target_index)
							moved = True
							break

					if not moved:
						delta.removed_trap_redistributions[index].append(
							random.sample(
								delta.valid_trap_indexes & modified,
								1
							)[0]
						)

		node_moves = {}

		for from_indexes, to_index in random.sample(moves.items(), random.randint(min_moves, max_moves)):
			if len(modified) >= delta.max_trap_difference:
				break

			if len(modified | {from_indexes[0], to_index}) > delta.max_trap_difference:
				continue

			node_moves[from_indexes] = to_index
			modified.add(from_indexes[0])
			modified.add(to_index)

		delta.node_moves = node_moves

		added_nodes = {}

		for added_node in delta.added_node_indexes:
			possible_indexes = adds[added_node]

			if len(modified) >= delta.max_trap_difference:
				possible_indexes -= modified

				if not possible_indexes:
					possible_indexes = modified

			index = random.sample(possible_indexes, 1)[0]
			added_nodes[added_node] = index
			modified.add(index)

		for index in delta.removed_trap_redistributions:
			delta.evacuate_removed_trap(index)

	return delta_1, delta_2


def logistic(x: float, max_value: float, mid: float, slope: float) -> float:
	return max_value / (1 + math.e ** (slope * (x - mid)))


class Constraint(ABC):
	description = 'A constraint' #type: str

	def __init__(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		self._constrained_nodes = constrained_nodes
		self._trap_amount = trap_amount

	@abstractmethod
	def score(self, distribution: TrapDistribution) -> float:
		pass


class ConstraintSet(object):

	def __init__(self, constraints: t.Tuple[t.Tuple[Constraint, float], ...]):
		self._constraints = constraints
		self._weights = tuple(weight for _, weight in self._constraints)

	def score(self, distribution: TrapDistribution) -> t.Tuple[float, ...]:
		unweighted_values = tuple(
			constraint.score(distribution)
			for constraint, _ in
			self._constraints
		)

		return (
			functools.reduce(
				operator.mul,
				(
					value ** weight
					for value, weight in
					zip(unweighted_values, self._weights)
				)
			),
		) + unweighted_values

	def total_score(self, distribution: TrapDistribution) -> float:
		return self.score(distribution)[0]

	def __iter__(self) -> t.Iterable[Constraint]:
		return (constraint for constraint, _ in self._constraints)
	
	
class ConstraintSetBluePrint(object):

	def __init__(self, *constraints: t.Tuple[t.Type[Constraint], float, t.Dict]):
		self._constraints = constraints
		
	def realise(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	) -> ConstraintSet:
		return ConstraintSet(
			tuple(
				(
					constraint_type(constrained_nodes, trap_amount, random_population, **kwargs),
					weight,
				)
				for constraint_type, weight, kwargs in
				self._constraints
			)
		)


class ValueDistributionHomogeneityConstraint(Constraint):
	description = 'Value distribution homogeneity'

	def __init__(
		self,
		nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		super().__init__(nodes, trap_amount, random_population)
		self._average_trap_value = (
			sum(
				(
					cubeable.value
					for cubeable in
					self._constrained_nodes
				)
			) / self._trap_amount
		)

		self._average_value_distribution_homogeneity_factor = statistics.mean(
			self._value_distribution_heterogeneity_factor(
				distribution
			)
			for distribution in
			random_population
		)

	def _value_distribution_heterogeneity_factor(self, distribution: TrapDistribution) -> float:
		return sum(
			(
				abs(
					sum(
						cubeable.value
						for cubeable in
						trap
					) - self._average_trap_value
				) ** 2
				for trap in
				distribution.traps
			)
		)

	def _value_distribution_homogeneity_factor(self, distribution: TrapDistribution) -> float:
		return self._value_distribution_heterogeneity_factor(
			distribution
		) / self._average_value_distribution_homogeneity_factor

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._value_distribution_homogeneity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class GroupExclusivityConstraint(Constraint):
	description = 'Group exclusivity'

	def __init__(
		self,
		nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
		group_weights: t.Dict[str, float],
	):
		super().__init__(nodes, trap_amount, random_population)
		self._group_weights = {} if group_weights is None else group_weights

		self._average_group_collision_factor = statistics.mean(
			self._group_collision_factor(
				distribution
			)
			for distribution in
			random_population
		)
		
	def _get_group_weight(self, group: str) -> float:
		return self._group_weights.get(group, 1.)

	def _group_collision_factor(self, distribution: TrapDistribution) -> int:
		collision_factor = 0

		for trap in distribution.traps:
			groups = {}  # type: t.Dict[str, t.List[ConstrainedNode]]
			collisions = {}  # type: t.Dict[t.FrozenSet[ConstrainedNode], t.List[str]]

			for constrained_node in trap:
				for group in constrained_node.groups:

					if group in groups:
						for other_node in groups[group]:
							_collision_key = frozenset((constrained_node, other_node))
							try:
								collisions[_collision_key].append(group)
							except KeyError:
								collisions[_collision_key] = [group]

						groups[group].append(constrained_node)

					else:
						groups[group] = [constrained_node]

			collision_factor += sum(
				(
					sum(node.value for node in nodes) + 30
					* max(self._get_group_weight(group) for group in groups)
				) ** 1.5
				for nodes, groups in
				collisions.items()
			)

		return collision_factor

	def _group_exclusivity_factor(self, distribution: TrapDistribution) -> float:
		return self._group_collision_factor(
			distribution
		) / (self._average_group_collision_factor + 1)

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._group_exclusivity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class SizeHomogeneityConstraint(Constraint):
	description = 'Size homogeneity'

	def __init__(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		super().__init__(constrained_nodes, trap_amount, random_population)
		self._average_trap_size = len(self._constrained_nodes) / self._trap_amount

		self._average_size_homogeneity_factor = statistics.mean(
			self._size_heterogeneity_factor(
				distribution
			)
			for distribution in
			random_population
		)

	def _size_heterogeneity_factor(self, distribution: TrapDistribution) -> float:
		return sum(
			abs(
				len(trap) - self._average_trap_size
			) ** 2
			for trap in
			distribution.traps
		)

	def _size_homogeneity_factor(self, distribution: TrapDistribution) -> float:
		return self._size_heterogeneity_factor(
			distribution
		) / self._average_size_homogeneity_factor

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._size_homogeneity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class Distributor(object):

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode],
		trap_amount: int,
		constraint_set_blue_print: ConstraintSetBluePrint,
		mate_chance: float = .5,
		mutate_chance: float = .2,
		tournament_size: int = 3,
		population_size: int = 300,
	):
		self._constrained_nodes = frozenset(constrained_nodes)
		self._trap_amount = trap_amount
		self._constraint_set_blue_print = constraint_set_blue_print
		self._mate_chance = mate_chance
		self._mutate_chance = mutate_chance
		self._tournament_size = tournament_size
		self._population_size = population_size

		self._toolbox = base.Toolbox()

		self._initialize_toolbox(self._toolbox)

		self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)

		self._population = self._toolbox.population(n=self._population_size) #type: t.List[TrapDistribution]

		self._sample_random_population = [
			TrapDistribution(self._constrained_nodes, self._trap_amount, random_initialization=True)
			for _ in
			range(self._population_size)
		]

		self._constraint_set = self._constraint_set_blue_print.realise(
			self._constrained_nodes,
			self._trap_amount,
			self._sample_random_population,
		)

		self._toolbox.register('select', tools.selTournament, tournsize=self._tournament_size)

		self._best = None

		self._statistics = tools.Statistics(key=lambda ind: ind)
		self._statistics.register('avg', lambda s: statistics.mean(e.fitness.values[0] for e in s))
		self._statistics.register('max', lambda s: max(e.fitness.values[0] for e in s))

		class _MaxMap(object):

			def __init__(self, _index: int):
				self._index = _index

			def __call__(self, population: t.Collection[TrapDistribution]):
				return sorted(
					(
						individual
						for individual in
						population
					),
					key = lambda individual:
						individual.fitness.values[0]
				)[-1].fitness.values[self._index]

		for index, constraint in enumerate(self._constraint_set):
			self._statistics.register(
				constraint.description,
				_MaxMap(index + 1),
			)

		self._logbook = None #type: tools.Logbook

	def _initialize_toolbox(self, toolbox: base.Toolbox):
		toolbox.register(
			'individual',
			TrapDistribution,
			constrained_nodes = self._constrained_nodes,
			trap_amount = self._trap_amount,
			random_initialization = True,
		)

		toolbox.register('evaluate', lambda d: self._constraint_set.score(d))
		toolbox.register('mate', mate_distributions, distributor=self)
		toolbox.register('mutate', mutate_trap_distribution)

	@property
	def population(self) -> t.List[TrapDistribution]:
		return self._population

	@property
	def sample_random_population(self) -> t.List[TrapDistribution]:
		return self._sample_random_population

	@property
	def best(self) -> TrapDistribution:
		if self._best is None:
			self._best = tools.selBest(self._population, 1)[0]

		return self._best

	@property
	def constrained_nodes(self) -> t.FrozenSet[ConstrainedNode]:
		return self._constrained_nodes

	@property
	def trap_amount(self) -> int:
		return self._trap_amount

	@property
	def constraint_set(self) -> ConstraintSet:
		return self._constraint_set

	@classmethod
	def trap_collection_to_trap_distribution(
		cls,
		traps: t.Collection[Trap],
		constrained_nodes: t.Iterable[ConstrainedNode],
	) -> t.Tuple[TrapDistribution, t.List[ConstrainedNode], t.List[t.Tuple[PrintingNode, int]]]:

		constraint_map = {} #type: t.Dict[PrintingNode, t.List[ConstrainedNode]]

		for constrained_node in constrained_nodes:
			try:
				constraint_map[constrained_node.node].append(constrained_node)
			except KeyError:
				constraint_map[constrained_node.node] = [constrained_node]

		distribution = [[] for _ in range(len(traps))] #type: t.List[t.List[ConstrainedNode]]
		removed = [] #type: t.List[t.Tuple[PrintingNode, int]]

		for index, trap in enumerate(traps):

			for child in trap.node.children:
				printing_node = child if isinstance(child, PrintingNode) else AllNode((child,))

				try:
					distribution[index].append(constraint_map[printing_node].pop())
				except (KeyError, IndexError):
					removed.append((printing_node, index))

		added = list(
			itertools.chain(
				*(
					nodes
					for nodes in
					constraint_map.values()
				)
			)
		)

		return TrapDistribution(traps=distribution), added, removed

	def evaluate_cube(self, traps: t.Collection[Trap]) -> float:
		distribution, added, removed = self.trap_collection_to_trap_distribution(
			traps,
			self._constrained_nodes,
		)
		if added or removed:
			raise ValueError(f'Collection does not match distribution. Added: "{added}", removed: "{removed}".')

		return self._constraint_set.score(distribution)[0]

	def show_plot(self) -> 'Distributor':
		generations = self._logbook.select("gen")
		fit_maxes = self._logbook.select('max')
		fit_averages = self._logbook.select('avg')

		fig, ax1 = plt.subplots()

		colors = ('r', 'g', 'b', 'c', 'm', 'y')

		lines = functools.reduce(
			operator.add,
			(
				ax1.plot(
					generations,
					self._logbook.select(constraint.description),
					color,
					label = f'Max {constraint.description} score',
				)
				for constraint, color in
				zip(self._constraint_set, colors)
			),
		)

		ax2 = ax1.twinx()

		max_line = ax2.plot(generations, fit_maxes, 'k', label='Maximum Fitness')
		average_line = ax2.plot(generations, fit_averages, '.75', label='Average Fitness')

		lns = max_line + average_line + lines
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc="lower right")

		plt.show()

		return self

	def evaluate(self, generations: int) -> 'Distributor':
		population, logbook = algorithms.eaSimple(
			self._population,
			self._toolbox,
			self._mate_chance,
			self._mutate_chance,
			generations,
			stats = self._statistics,
		)

		self._population = population
		self._logbook = logbook
		self._best = None

		return self


class DeltaDistributor(Distributor):

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode],
		origin_trap_collection: t.Collection[Trap],
		constraint_set_blue_print: ConstraintSetBluePrint,
		max_trap_delta: int,
		trap_amount: t.Optional[int] = None,
		mate_chance: float = .5,
		mutate_chance: float = .2,
		tournament_size: int = 3,
		population_size: int = 300,
	):
		self._origin_trap_collection = origin_trap_collection
		self._max_trap_delta = max_trap_delta

		distribution, added, removed = self.trap_collection_to_trap_distribution(
			self._origin_trap_collection,
			constrained_nodes,
		)

		self._origin_trap_distribution = distribution
		self._added = added
		self._removed_trap_indexes = frozenset(
			index
			for node, index in
			removed
		)

		super().__init__(
			constrained_nodes,
			(
				len(origin_trap_collection)
				if trap_amount is None else
				trap_amount
			),
			constraint_set_blue_print,
			mate_chance,
			mutate_chance,
			tournament_size,
			population_size,
		)

	def _initialize_toolbox(self, toolbox: base.Toolbox):
		toolbox.register(
			'individual',
			DistributionDelta,
			origin = self._origin_trap_distribution,
			added_nodes = self._added,
			removed_node_indexes = self._removed_trap_indexes,
			max_trap_difference = self._max_trap_delta,
			trap_amount_delta = self.trap_amount - self._origin_trap_distribution.trap_amount,
		)

		toolbox.register('evaluate', lambda d: self._constraint_set.score(d.trap_distribution))
		toolbox.register('mate', mate_distribution_deltas)
		toolbox.register('mutate', mutate_distribution_delta)
